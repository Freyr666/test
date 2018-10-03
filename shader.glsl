#version 430
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_image_load_store : enable
#extension GL_ARB_shader_storage_buffer_object : enable

//precision lowp float;

#define WHT_LVL 0.90196078
// 210
#define BLK_LVL 0.15625
// 40
#define WHT_DIFF 0.0234375
// 6
#define GRH_DIFF 0.0078125
// 2
#define KNORM 4.0
#define L_DIFF 5

#define BLOCK_SIZE 8

#define BLOCKY 0
#define LUMA 1
#define BLACK 2
#define DIFF 3
#define FREEZE 4

struct Noize {
        float noize;
        float frozen;
        float black;
        float bright;
        float diff;
        int   visible;
};

layout (r8, binding = 0) uniform image2D tex;
layout (r8, binding = 1) uniform image2D tex_prev;
uniform int width;
uniform int height;
uniform int stride;
uniform int black_bound;
uniform int freez_bound;

layout (std430, binding=10) buffer Interm {
         Noize noize_data [];
};

//const uint wht_coef[20] = {8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 14, 17, 27};
//const uint ght_coef[20] = {4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 10, 13, 23};
const uint wht_coef[20] = {10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 16, 19, 29};	       
const uint ght_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};
//const uint wht_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};
//const uint ght_coef[20] = {2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 8, 11, 21};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in; 

float compute_noize (ivec2 pos) {
        float lvl;
        float pix       = imageLoad(tex, pos).r;
        float pix_right = imageLoad(tex, ivec2(pos.x + 1, pos.y)).r;
        float pix_down  = imageLoad(tex, ivec2(pos.x, pos.y + 1)).r;
        /* Noize */
        float res = 0.0;
        if ((pix < WHT_LVL) && (pix > BLK_LVL)) {
                lvl = GRH_DIFF;
        } else {
                lvl = WHT_DIFF;
        }
        if (abs(pix - pix_right) >= lvl) {
                res += 1.0/(8.0*8.0*2.0);
        }
        if (abs(pix - pix_down) >= lvl) {
                res += 1.0/(8.0*8.0*2.0);
        }
        return res;
}

void main() {
        uint block_pos = (gl_WorkGroupID.y * (width / BLOCK_SIZE)) + gl_WorkGroupID.x;
        /* Block init */

        noize_data[block_pos].noize = 0.0;
        noize_data[block_pos].black = 0.0;
        noize_data[block_pos].frozen = 0.0;
        noize_data[block_pos].bright = 0.0;
        noize_data[block_pos].diff = 0.0;
        noize_data[block_pos].visible = 0;
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                        float diff_loc;
                        ivec2 pix_pos = ivec2(gl_WorkGroupID.x * BLOCK_SIZE + i,
                                              gl_WorkGroupID.y * BLOCK_SIZE + j);
                        float pix = imageLoad(tex, pix_pos).r;
                        /* Noize */
                        noize_data[block_pos].noize += compute_noize(pix_pos);
                        /* Brightness */
                        noize_data[block_pos].bright += float(pix);
                        /* Black */
                        if (pix <= float(black_bound / 255.0)) {
                                noize_data[block_pos].black += 1.0;
                        }
                        /* Diff */
                        diff_loc = abs(pix - imageLoad(tex_prev, pix_pos).r);
                        noize_data[block_pos].diff += diff_loc;
                        /* Frozen */
                        if (diff_loc <= float(freez_bound / 255.0)) {
                                noize_data[block_pos].frozen += 1.0;
                        }
                }
        }
}
