#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "visualizer.h"
#include "player/core.h"
#include "audio/out/internal.h"
#include "osdep/timer.h"
#include "misc/fft_util.h"
#include "video/out/gpu/shader_cache.h"


static const struct ra_renderpass_input bar_vao[] = {
    {"position", RA_VARTYPE_FLOAT, 2, 1, offsetof(struct bar_vertex, position)},
    {"vcolor", RA_VARTYPE_FLOAT, 4, 1, offsetof(struct bar_vertex, color)},
};


static const struct ra_renderpass_input circle_vao[] = {
    {"position", RA_VARTYPE_FLOAT, 2, 1, offsetof(struct circle_vertex, position)},
    {"texcoord", RA_VARTYPE_FLOAT, 2, 1, offsetof(struct circle_vertex, texcoord)},
    {"vcolor", RA_VARTYPE_FLOAT, 4, 1, offsetof(struct circle_vertex, color)},
};


static float clampf(const float v, const float lo, const float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float maxf(const float a, const float b) { return a > b ? a : b; }
static size_t minz(const size_t a, const size_t b) { return a < b ? a : b; }

static void hsv_to_rgb(const float h, const float s, const float v,
                       float *r, float *g, float *b) {
    const float hh = h - floorf(h);
    const float sc = hh * 6.0f;
    const int sec = (int) sc;
    const float f = sc - (float) sec;
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));

    switch (sec % 6) {
        case 0: *r = v;
            *g = t;
            *b = p;
            break;
        case 1: *r = q;
            *g = v;
            *b = p;
            break;
        case 2: *r = p;
            *g = v;
            *b = t;
            break;
        case 3: *r = p;
            *g = q;
            *b = v;
            break;
        case 4: *r = t;
            *g = p;
            *b = v;
            break;
        default: *r = v;
            *g = p;
            *b = q;
            break;
    }
}

static void push_bar_quad(gpu_visualizer *vis,
                          const float x0, const float x1, const float y0, const float y1,
                          const float r, const float g, const float b, const float a) {
    if (vis->bar_count + 6 > MAX_BAR_VERTS) return;
    struct bar_vertex *v = &vis->bar_verts[vis->bar_count];
    v[0] = (struct bar_vertex){{x0, y0}, {r, g, b, a}};
    v[1] = (struct bar_vertex){{x1, y0}, {r, g, b, a}};
    v[2] = (struct bar_vertex){{x0, y1}, {r, g, b, a}};
    v[3] = (struct bar_vertex){{x0, y1}, {r, g, b, a}};
    v[4] = (struct bar_vertex){{x1, y0}, {r, g, b, a}};
    v[5] = (struct bar_vertex){{x1, y1}, {r, g, b, a}};
    vis->bar_count += 6;
}

static void push_circle_quad(gpu_visualizer *vis,
                             const float x, const float y, const float size_ndc,
                             const float r, const float g, const float b, const float a) {
    if (vis->circle_count + 6 > MAX_CIRCLE_VERTS || vis->height <= 0) return;
    const float aspect = (float) vis->width / (float) vis->height;
    const float hsw = size_ndc * 0.5f;
    const float hsh = hsw * aspect;
    const float x0 = x - hsw;
    const float x1 = x + hsw;
    const float y0 = y - hsh;
    const float y1 = y + hsh;
    struct circle_vertex *v = &vis->circle_verts[vis->circle_count];
    v[0] = (struct circle_vertex){{x0, y0}, {0, 0}, {r, g, b, a}};
    v[1] = (struct circle_vertex){{x1, y0}, {1, 0}, {r, g, b, a}};
    v[2] = (struct circle_vertex){{x0, y1}, {0, 1}, {r, g, b, a}};
    v[3] = (struct circle_vertex){{x0, y1}, {0, 1}, {r, g, b, a}};
    v[4] = (struct circle_vertex){{x1, y0}, {1, 0}, {r, g, b, a}};
    v[5] = (struct circle_vertex){{x1, y1}, {1, 1}, {r, g, b, a}};
    vis->circle_count += 6;
}

gpu_visualizer *gpu_visualizer_create(struct ra *ra, struct mpv_global *global, struct mp_log *log) {
    gpu_visualizer *vis = talloc_zero(NULL, struct gpu_visualizer);
    if (!vis) return NULL;
    vis->ra = ra;
    vis->sc = gl_sc_create(ra, global, log);
    return vis;
}

void gpu_visualizer_destroy(gpu_visualizer **gpu_pvis) {
    if (!gpu_pvis || !*gpu_pvis) return;
    gpu_visualizer *vis = *gpu_pvis;
    gl_sc_destroy(vis->sc);
    talloc_free(vis);
    *gpu_pvis = NULL;
}

void gpu_visualizer_resize(gpu_visualizer *vis, const int w, const int h) {
    if (!vis) return;
    vis->width = w;
    vis->height = h;
}

static size_t fft_analyze(const float dt, const MPContext *mpctx) {
    struct mp_visualizer *visualizer = mpctx->visualizer;

    const int FFT_SIZE = visualizer->pcm_buffer_size;
    if (mpctx->ao_chain && mpctx->ao_chain->ao)
        visualizer->sample_rate = mpctx->ao_chain->ao->samplerate;
    if (visualizer->sample_rate == 0) return 0;

    for (size_t i = 0; i < FFT_SIZE; ++i) {
        const float t = (float) i / (float) (FFT_SIZE - 1);
        const float hann = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * t);
        visualizer->in_win[i] = visualizer->in_raw[i] * hann;
    }

    fft_compute(visualizer->in_win, visualizer->out_raw, FFT_SIZE);

    const float lowf = 1.0f;
    size_t m = 0;
    float max_amp = 1.0f;
    float f = lowf;
    const float max_f = 16000;
    if (visualizer->sample_rate <= 0) return 0;
    const size_t cutoff = minz(FFT_SIZE / 2, (size_t) (max_f * (float) FFT_SIZE / (float) visualizer->sample_rate));

    while ((size_t) f < cutoff && m < FFT_SIZE) {
        const float step = 1.06f;
        const float f1 = ceilf(f * step);
        float a = 0.0f;
        for (size_t q = f; q < cutoff && q < (size_t) f1; ++q) {
            const float b = fft_amp(visualizer->out_raw[q]);
            if (b > a) a = b;
        }
        if (max_amp < a) max_amp = a;
        visualizer->out_log[m++] = a;
        f = f1;
    }
    for (size_t i = 0; i < m; ++i) visualizer->out_log[i] /= max_amp;

    for (size_t i = 0; i < m; ++i) {
        visualizer->out_smooth[i] += (visualizer->out_log[i] - visualizer->out_smooth[i]) * 8.0f * dt;
        visualizer->out_smear[i] += (visualizer->out_smooth[i] - visualizer->out_smear[i]) * 3.0f * dt;
    }
    return m;
}

bool gpu_visualizer_draw(gpu_visualizer *vis, MPContext *mpctx, struct ra_tex *target) {
    if (!vis || !mpctx || !mpctx->visualizer || vis->width <= 0) return false;

    const int64_t now = mp_time_ns();
    float dt = 1.0f / 60.0f;
    if (vis->last_render_ns) {
        const int64_t elapsed = now - vis->last_render_ns;
        dt = clampf((float) elapsed / 1e9f, 1.0f / 240.0f, 0.1f);
    }
    vis->last_render_ns = now;

    const size_t bins = fft_analyze(dt, mpctx);

    vis->bar_count = 0;
    vis->circle_count = 0;
    const float cell = 2.0f / (float) VIS_BARS;

    for (size_t i = 0; i < VIS_BARS; ++i) {
        const size_t si = minz(bins - 1, (i * bins) / VIS_BARS);
        const float smooth = clampf(mpctx->visualizer->out_smooth[si], 0.0f, 1.0f);
        float cr, cg, cb;
        hsv_to_rgb((float) i / (float) VIS_BARS, 0.75f, 1.0f, &cr, &cg, &cb);
        const float xc = -1.0f + ((float) i + 0.5f) * cell;
        const float hw = cell * 0.08f;
        const float yb = -1.0f;
        const float ys = yb + smooth * VIS_HEIGHT;
        push_bar_quad(vis, xc - hw, xc + hw, yb, ys, cr, cg, cb, 1.0f);
    }

    for (size_t i = 0; i < VIS_BARS; ++i) {
        const size_t si = minz(bins - 1, (i * bins) / VIS_BARS);
        const float smooth = clampf(mpctx->visualizer->out_smooth[si], 0.0f, 1.0f);
        const float smear = clampf(mpctx->visualizer->out_smear[si], 0.0f, 1.0f);
        float cr, cg, cb;
        hsv_to_rgb((float) i / (float) VIS_BARS, 0.75f, 1.0f, &cr, &cg, &cb);
        const float xc = -1.0f + ((float) i + 0.5f) * cell;
        const float ys = -1.0f + smooth * VIS_HEIGHT;
        const float ym = -1.0f + smear * VIS_HEIGHT;
        const float dy = ym - ys;
        const float dist = fabsf(dy);
        int steps = (int) (dist * 28.0f);
        if (steps < 3) steps = 3;
        if (steps > 8) steps = 8;
        for (int s = 0; s < steps; ++s) {
            const float u = (float) (s + 1) / (float) (steps + 1);
            const float ty = ys + dy * u;
            const float fo = 1.0f - u;
            const float rn = maxf(cell * 0.08f, cell * 2.2f * sqrtf(maxf(smooth, 1e-4f)) * (0.45f + 0.55f * fo));
            push_circle_quad(vis, xc, ty, rn * 2.0f, cr, cg, cb, 0.18f + 0.30f * fo);
        }
    }

    const int smear_count = vis->circle_count;
    for (size_t i = 0; i < VIS_BARS; ++i) {
        const size_t si = minz(bins - 1, (i * bins) / VIS_BARS);
        const float smooth = clampf(mpctx->visualizer->out_smooth[si], 0.0f, 1.0f);
        float cr, cg, cb;
        hsv_to_rgb((float) i / (float) VIS_BARS, 0.75f, 1.0f, &cr, &cg, &cb);
        const float xc = -1.0f + ((float) i + 0.5f) * cell;
        const float ys = -1.0f + smooth * VIS_HEIGHT;
        const float cn = maxf(cell * 0.2f, cell * 3.0f * sqrtf(smooth));
        push_circle_quad(vis, xc, ys, cn * 2.0f, cr, cg, cb, 0.9f);
    }

    gl_sc_reset(vis->sc);
    gl_sc_blend(vis->sc, RA_BLEND_SRC_ALPHA, RA_BLEND_ONE_MINUS_SRC_ALPHA, RA_BLEND_ONE, RA_BLEND_ONE_MINUS_SRC_ALPHA);

    /* Draw bars */
    gl_sc_add(vis->sc, "color = vcolor;");
    gl_sc_dispatch_draw(vis->sc, target, false, bar_vao, 2, sizeof(struct bar_vertex), vis->bar_verts, vis->bar_count);

    /* Draw circles */
    gl_sc_reset(vis->sc);
    gl_sc_blend(vis->sc, RA_BLEND_SRC_ALPHA, RA_BLEND_ONE_MINUS_SRC_ALPHA, RA_BLEND_ONE, RA_BLEND_ONE_MINUS_SRC_ALPHA);
    gl_sc_uniform_f(vis->sc, "u_radius", 0.30f);
    gl_sc_uniform_f(vis->sc, "u_power", 3.0f);
    gl_sc_add(vis->sc, "vec2 p = texcoord - vec2(0.5);\n"
              "float d = length(p);\n"
              "if (d > 0.5) discard;\n"
              "float s = d - u_radius;\n"
              "if (s <= 0.0) {\n"
              "  color = vcolor * 1.5;\n"
              "} else {\n"
              "  float denom = max(0.0001, 0.5 - u_radius);\n"
              "  float t = clamp(1.0 - s / denom, 0.0, 1.0);\n"
              "  color = mix(vec4(vcolor.xyz, 0.0), vcolor * 1.5, pow(t, u_power));\n"
              "}\n");
    if (smear_count > 0) {
        gl_sc_dispatch_draw(vis->sc, target, false, circle_vao, 3, sizeof(struct circle_vertex), vis->circle_verts,
                            smear_count);
    }

    /* Tip circles */
    if (vis->circle_count > smear_count) {
        gl_sc_blend(vis->sc, RA_BLEND_SRC_ALPHA, RA_BLEND_ONE_MINUS_SRC_ALPHA, RA_BLEND_ONE,
                    RA_BLEND_ONE_MINUS_SRC_ALPHA);
        gl_sc_uniform_f(vis->sc, "u_radius", 0.07f);
        gl_sc_uniform_f(vis->sc, "u_power", 5.0f);
        gl_sc_add(vis->sc, "vec2 p = texcoord - vec2(0.5);\n"
                  "float d = length(p);\n"
                  "if (d > 0.5) discard;\n"
                  "float s = d - u_radius;\n"
                  "if (s <= 0.0) {\n"
                  "  color = vcolor * 1.5;\n"
                  "} else {\n"
                  "  float denom = max(0.0001, 0.5 - u_radius);\n"
                  "  float t = clamp(1.0 - s / denom, 0.0, 1.0);\n"
                  "  color = mix(vec4(vcolor.xyz, 0.0), vcolor * 1.5, pow(t, u_power));\n"
                  "}\n");
        gl_sc_dispatch_draw(vis->sc, target, false, circle_vao, 3, sizeof(struct circle_vertex),
                            &vis->circle_verts[smear_count], vis->circle_count - smear_count);
    }

    return true;
}
