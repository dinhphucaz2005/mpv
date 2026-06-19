#ifndef MP_GPU_VISUALIZER_H
#define MP_GPU_VISUALIZER_H

#include "ra.h"
#include "shader_cache.h"

struct circle_vertex {
    float position[2];
    float texcoord[2];
    float color[4];
};

struct bar_vertex {
    float position[2];
    float color[4];
};

#define VIS_BARS         64
#define VIS_HEIGHT       (4.0f / 3.0f)
#define MAX_BAR_VERTS    (VIS_BARS * 6)
#define MAX_CIRCLE_VERTS (VIS_BARS * 10 * 6)

struct gpu_visualizer {
    struct ra *ra;
    struct gl_shader_cache *sc;

    int width, height;
    __int64_t last_render_ns;

    struct bar_vertex bar_verts[MAX_BAR_VERTS];
    int bar_count;
    struct circle_vertex circle_verts[MAX_CIRCLE_VERTS];
    int circle_count;
};

typedef struct gpu_visualizer gpu_visualizer;

gpu_visualizer *gpu_visualizer_create(struct ra *ra, struct mpv_global *global, struct mp_log *log);

void gpu_visualizer_destroy(gpu_visualizer **gpu_pvis);

void gpu_visualizer_resize(gpu_visualizer *vis, int w, int h);

bool gpu_visualizer_draw(gpu_visualizer *vis, struct MPContext *mpctx, struct ra_tex *target);

#endif
