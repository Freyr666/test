#version 430
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_image_load_store : enable
#extension GL_ARB_shader_storage_buffer_object : enable

precision lowp float;

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
        float frozen;
        float black;
        float bright;
        float diff;
        int   visible;
};

layout (r8, binding = 0) uniform lowp image2D tex;
layout (r8, binding = 1) uniform lowp image2D tex_prev;
uniform int width;
uniform int height;
uniform int stride;
uniform int black_bound;
uniform int freez_bound;

layout (std430, binding=10) buffer Interm {
         Noize noize_data [];
};

layout (std430, binding=11) buffer Result {
        float data [5];
};

//const uint wht_coef[20] = {8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 14, 17, 27};
//const uint ght_coef[20] = {4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 10, 13, 23};
const uint wht_coef[20] = {10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 16, 19, 29};	       
const uint ght_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};
//const uint wht_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};
//const uint ght_coef[20] = {2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 8, 11, 21};

layout (local_size_x = 1, local_size_y = 1) in;

float get_coef(float noize, uint array[20]) {
        uint ret_val;                                     
        if((noize>100) || (noize<0))
                ret_val = 0;                              
        else                                              
                ret_val = array[uint(noize/5)];                 
        return float(ret_val/255.0);
}
       

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
        uint block_pos = (gl_GlobalInvocationID.y * (width / BLOCK_SIZE)) + gl_GlobalInvocationID.x ;
        /* Block init */
        float noize = 0.0;
        /* l r u d */
        vec4 noize_v = vec4(0);
        
        noize_data[block_pos].black = 0.0;
        noize_data[block_pos].frozen = 0.0;
        noize_data[block_pos].bright = 0.0;
        noize_data[block_pos].diff = 0.0;
        noize_data[block_pos].visible = 0;
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                        float diff_loc;
                        ivec2 pix_pos = ivec2(gl_GlobalInvocationID.x * BLOCK_SIZE + i,
                                              gl_GlobalInvocationID.y * BLOCK_SIZE + j);
                        float pix = imageLoad(tex, pix_pos).r;
                        /* Noize */
                        noize += compute_noize(pix_pos);
                        if (gl_GlobalInvocationID.x > 0)
                                noize_v[0] += compute_noize(ivec2(pix_pos.x - BLOCK_SIZE, pix_pos.y));
                        if (gl_GlobalInvocationID.y > 0)
                                noize_v[3] += compute_noize(ivec2(pix_pos.x, pix_pos.y - BLOCK_SIZE));
                        if (gl_GlobalInvocationID.x < (width / BLOCK_SIZE))
                                noize_v[1] += compute_noize(ivec2(pix_pos.x + BLOCK_SIZE, pix_pos.y));
                        if (gl_GlobalInvocationID.x < (height / BLOCK_SIZE))
                                noize_v[2] += compute_noize(ivec2(pix_pos.x, pix_pos.y + BLOCK_SIZE));
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
        /* Noize coeffs */
        noize_v = 100.0 * vec4(max(noize_v[0], noize),
                               max(noize_v[1], noize),
                               max(noize_v[2], noize),
                               max(noize_v[3], noize));
        vec4 white_v = vec4(get_coef(noize_v[0], wht_coef),
                            get_coef(noize_v[1], wht_coef),
                            get_coef(noize_v[2], wht_coef),
                            get_coef(noize_v[3], wht_coef));
        vec4 grey_v = vec4(get_coef(noize_v[0], ght_coef),
                           get_coef(noize_v[1], ght_coef),
                           get_coef(noize_v[2], ght_coef),
                           get_coef(noize_v[3], ght_coef));
        /* compute borders */
        if (gl_GlobalInvocationID.x == 0
            || gl_GlobalInvocationID.x >= (width / BLOCK_SIZE)
            || gl_GlobalInvocationID.y == 0
            || gl_GlobalInvocationID.y >= (height / BLOCK_SIZE))
                return;
        ivec4 vis = {0,0,0,0};
        for (int i = 0; i < BLOCK_SIZE; i++) {
                /* l r u d */
                ivec2 zero = ivec2(gl_GlobalInvocationID.x * BLOCK_SIZE,
                                   gl_GlobalInvocationID.y * BLOCK_SIZE);
                vec4 pixel = vec4(imageLoad(tex, ivec2(zero.x, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x+BLOCK_SIZE-1, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y+BLOCK_SIZE-1)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y)).r );
                vec4 prev  = vec4(imageLoad(tex, ivec2(zero.x + 1, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x+BLOCK_SIZE-2, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y+BLOCK_SIZE-2)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y + 1)).r );
                vec4 next  = vec4(imageLoad(tex, ivec2(zero.x - 1, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x+BLOCK_SIZE, zero.y + i)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y+BLOCK_SIZE)).r,
                                  imageLoad(tex, ivec2(zero.x + i, zero.y - 1)).r );
                vec4 next_next  = vec4(imageLoad(tex, ivec2(zero.x - 2, zero.y + i)).r,
                                       imageLoad(tex, ivec2(zero.x+BLOCK_SIZE+1, zero.y + i)).r,
                                       imageLoad(tex, ivec2(zero.x + i, zero.y+BLOCK_SIZE+1)).r,
                                       imageLoad(tex, ivec2(zero.x + i, zero.y - 2)).r );
                vec4 coef = vec4((pixel[0] < WHT_LVL) && (pixel[0] > BLK_LVL) ?
                                 grey_v[0] : white_v[0],
                                 (pixel[1] < WHT_LVL) && (pixel[1] > BLK_LVL) ?
                                 grey_v[1] : white_v[1],
                                 (pixel[2] < WHT_LVL) && (pixel[2] > BLK_LVL) ?
                                 grey_v[2] : white_v[2],
                                 (pixel[3] < WHT_LVL) && (pixel[3] > BLK_LVL) ?
                                 grey_v[3] : white_v[3]);
                vec4 denom = round( (abs(prev-pixel) + abs(next-next_next)) / KNORM);
                denom = vec4(denom[0] == 0.0 ? 1.0 : denom[0],
                             denom[1] == 0.0 ? 1.0 : denom[1],
                             denom[2] == 0.0 ? 1.0 : denom[2],
                             denom[3] == 0.0 ? 1.0 : denom[3]);
                vec4 norm = abs(next-pixel) / denom;
                vis += ivec4( norm[0] > coef[0] ? 1 : 0,
                              norm[1] > coef[1] ? 1 : 0,
                              norm[2] > coef[2] ? 1 : 0,
                              norm[3] > coef[3] ? 1 : 0 );
        }
        /* counting visible blocks */
        int loc_counter = 0;
        for (int side = 0; side < 4; side++) {
                if (vis[side] > L_DIFF)
                        loc_counter += 1;
        }
        if (loc_counter >= 2)
                noize_data[block_pos].visible = 1;
}
