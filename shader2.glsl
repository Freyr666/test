#version 430
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_image_load_store : enable
#extension GL_ARB_shader_storage_buffer_object : enable

#ifdef GL_ES
precision mediump float;
#endif

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

uniform int width;
uniform int height;

layout (std430, binding=10) buffer Interm {
         Noize noize_data [];
};

layout (std430, binding=11) buffer Result {
        float data [5];
};

layout (local_size_x = 1, local_size_y = 1) in;

void main() {
        uint block_pos = (gl_GlobalInvocationID.y * (width / BLOCK_SIZE)) + gl_GlobalInvocationID.x;

        if (gl_GlobalInvocationID.xy != vec2(0,0))
                return;
        
        float bright = 0.0;
        float diff = 0.0;
        float frozen = 0.0;
        float black  = 0.0;
        float blocky = 0.0;  
        for (int i = 0; i < (height * width / 64); i++) {
                bright += float(noize_data[i].bright);
                diff   += noize_data[i].diff;
                frozen += noize_data[i].frozen;
                black  += noize_data[i].black;
                if (noize_data[i].visible == 1)
                        blocky += 1.0;
        }
        data[LUMA] = 256.0 * bright / float(height * width);
        data[DIFF] = 100.0 * diff / float(height * width);
        data[BLACK] = 100.0 * black / float(height * width);
        data[FREEZE] = 100.0 * frozen / float(height * width);
        data[0] = 100.0 * blocky / float(height * width / 64);
}
