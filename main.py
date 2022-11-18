import pygame
import glm
import random
import numpy as np
from obj import * 
from OpenGL.GL import *
from OpenGL.GL.shaders import * 

pygame.init()

screen = pygame.display.set_mode(
    (800, 600),
    pygame.OPENGL | pygame.DOUBLEBUF
)

model = Obj('./cube2.obj')

vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;

uniform mat4 matrix;

out vec3 ourColor;
out vec2 fragCoord;

void main()
{
    gl_Position = matrix * vec4(position, 1.0f);
    ourColor = vertexColor;
    fragCoord = gl_Position.xy;
}
"""

fragment_shader = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;

in vec3 ourColor;

void main()
{
    //fragColor = vec4(color, 1.0f);
    fragColor = vec4(ourColor, 1.0f);
}
"""

fragment_shader2 = """
#version 460

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;



uniform float time;

void main()
{
    vec2 resolution = vec2(2, 2);
    vec3 c;
    float l, z= time;
    for(int i=0; i<3; i++) {
        vec2 uv,p=fragCoord.xy/resolution;
		uv=p;
		p-=.5;
		p.x*=resolution.x/resolution.y;
		z+=.07;
		l=length(p);
		uv+=p/l*(sin(z)+1.)*abs(sin(l*9.-z-z));
		c[i]=.01/length(mod(uv,1.)-.5);
    }
    fragColor=vec4(c/l,time);
    //fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

fragment_shader3 = """
#version 460

precision highp float;

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;

uniform float time;


mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c,s,-s,c);
}

const float pi = acos(-1.0);
const float pi2 = pi*2.0;

vec2 pmod(vec2 p, float r) {
    float a = atan(p.x, p.y) + pi/r;
    float n = pi2 / r;
    a = floor(a/n)*n;
    return p*rot(-a);
}

float box( vec3 p, vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float ifsBox(vec3 p) {
    for (int i=0; i<5; i++) {
        p = abs(p) - 1.0;
        p.xy *= rot(time*0.3);
        p.xz *= rot(time*0.1);
    }
    p.xz *= rot(time);
    return box(p, vec3(0.4,0.8,0.3));
}

float map(vec3 p, vec3 cPos) {
    vec3 p1 = p;
    p1.x = mod(p1.x-5., 10.) - 5.;
    p1.y = mod(p1.y-5., 10.) - 5.;
    p1.z = mod(p1.z, 16.)-8.;
    p1.xy = pmod(p1.xy, 5.0);
    return ifsBox(p1);
}

void main() {

    
    vec2 resolution = vec2(2, 2);
    vec2 p = (fragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);

    vec3 cPos = vec3(0.0,0.0, -3.0 * time);

    vec3 cDir = normalize(vec3(0.0, 0.0, -1.0));
    vec3 cUp  = vec3(sin(time), 1.0, 0.0);
    vec3 cSide = cross(cDir, cUp);

    vec3 ray = normalize(cSide * p.x + cUp * p.y + cDir);

    float acc = 0.0;
    float acc2 = 0.0;
    float t = 0.0;
    for (int i = 0; i < 99; i++) {
        vec3 pos = cPos + ray * t;
        float dist = map(pos, cPos);
        dist = max(abs(dist), 0.02);
        float a = exp(-dist*3.0);
        if (mod(length(pos)+24.0*time, 30.0) < 3.0) {
            a *= 2.0;
            acc2 += a;
        }
        acc += a;
        t += dist * 0.5;
    }

    vec3 col = vec3(acc * 0.01, acc * 0.011 + acc2*0.002, acc * 0.012+ acc2*0.005);
    fragColor = vec4(col, 1.0 - t * 0.03);
}
"""

fragment_shader4 = """
#version 460
    const float SHAPE_SIZE = .618;
    const float CHROMATIC_ABBERATION = .01;
    const float ITERATIONS = 10.;
    const float INITIAL_LUMA = .5;

    const float PI = 3.14159265359;
    const float TWO_PI = 6.28318530718;

    layout (location = 0) out vec4 fragColor;
    in vec3 ourColor;
    in vec2 fragCoord;

    uniform float time;


    mat2 rotate2d(float _angle){
        return mat2(cos(_angle),-sin(_angle),
                    sin(_angle),cos(_angle));
    }

    float sdPolygon(in float angle, in float distance) {
    float segment = TWO_PI / 4.0;
    return cos(floor(.5 + angle / segment) * segment - angle) * distance;
    }

    float getColorComponent(in vec2 st, in float modScale, in float blur) {
        vec2 modSt = mod(st, 1. / modScale) * modScale * 2. - 1.;
        float dist = length(modSt);
        float angle = atan(modSt.x, modSt.y) + sin(time * .08) * 9.0;
        //dist = sdPolygon(angle, dist);
        //dist += sin(angle * 3. + time * .21) * .2 + cos(angle * 4. - time * .3) * .1;
        float shapeMap = smoothstep(SHAPE_SIZE + blur, SHAPE_SIZE - blur, sin(dist * 3.0) * .5 + .5);
        return shapeMap;
    }

    void main() {
        
        vec2 resolution = vec2(2, 2);
        float blur = .4 + sin(time * .52) * .2;

        vec2 st =
            (2.* fragCoord - resolution.xy)
            / min(resolution.x, resolution.y);
        vec2 origSt = st;
        st *= rotate2d(sin(time * .14) * .3);
        st *= (sin(time * .15) + 2.) * .3;
        st *= log(length(st * .428)) * 1.1;

        float modScale = 1.;

        vec3 color = vec3(0);
        float luma = INITIAL_LUMA;
        for (float i = 0.; i < ITERATIONS; i++) {
            vec2 center = st + vec2(sin(time * .12), cos(time * .13));
            //center += pow(length(center), 1.);
            vec3 shapeColor = vec3(
                getColorComponent(center - st * CHROMATIC_ABBERATION, modScale, blur),
                getColorComponent(center, modScale, blur),
                getColorComponent(center + st * CHROMATIC_ABBERATION, modScale, blur)        
            ) * luma;
            st *= 1.1 + getColorComponent(center, modScale, .04) * 1.2;
            st *= rotate2d(sin(time  * .05) * 1.33);
            color += shapeColor;
            color = clamp(color, 0., 1.);
    //        if (color == vec3(1)) break;
            luma *= .6;
            blur *= .63;
        }
        const float GRADING_INTENSITY = .4;
        vec3 topGrading = vec3(
            1. + sin(time * 1.13 * .3) * GRADING_INTENSITY,
            1. + sin(time * 1.23 * .3) * GRADING_INTENSITY,
            1. - sin(time * 1.33 * .3) * GRADING_INTENSITY
        );
        vec3 bottomGrading = vec3(
            1. - sin(time * 1.43 * .3) * GRADING_INTENSITY,
            1. - sin(time * 1.53 * .3) * GRADING_INTENSITY,
            1. + sin(time * 1.63 * .3) * GRADING_INTENSITY
        );
        float origDist = length(origSt);
        vec3 colorGrading = mix(topGrading, bottomGrading, origDist - .5);
        fragColor = vec4(pow(color.rgb, colorGrading), 1.);
        fragColor *= smoothstep(2.1, .7, origDist);
}
"""

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
compiled_fragment_shader2 = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
compiled_fragment_shader3 = compileShader(fragment_shader3, GL_FRAGMENT_SHADER)
compiled_fragment_shader4 = compileShader(fragment_shader4, GL_FRAGMENT_SHADER)

shader = compileProgram(
    compiled_vertex_shader, 
    compiled_fragment_shader
)


shader2 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader2
)

shader3 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader3
)

shader4 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader4
)

glUseProgram(shader2)

vertex = []
for ver in model.vertices:
    for v in ver:
        vertex.append(v)

vertex_data = np.array(vertex, dtype=np.float32)



vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(
    GL_ARRAY_BUFFER, # tipo de datos
    vertex_data.nbytes, # tamaño de los datos en bytes
    vertex_data, # puntero a la data
    GL_STATIC_DRAW # tipo de uso de la data
)

glVertexAttribPointer(
    0, 
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(0)
)

glEnableVertexAttribArray(0)

faces = []
for face in model.faces:
    for f in face:
        faces.append(int(f[0])-1)

faces_data = np.array(faces, dtype=np.int32)


element_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_data.nbytes, faces_data, GL_STATIC_DRAW)



def calculateMatrix(angle, vector):

    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), vector)
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0, 0, 5),
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, 0)
    )

    projection = glm.perspective(
        glm.radians(45),
        1600 / 1200,
        0.1,
        1000
    )

    glViewport(0, 0, 800, 600)

    matrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, "matrix"),
        1,
        GL_FALSE,
        glm.value_ptr(matrix)
    )

running = True

glClearColor(0.0, 0.0, 0.0, 1.0)
r = 0
changeShader = False
shaderIndex = 0
shaders = [shader2, shader3, shader4, shader]
currentShader = shader2
vector = glm.vec3(0, 1, 0)

prev_time = pygame.time.get_ticks()

while running:
    glClear(GL_COLOR_BUFFER_BIT)

    if changeShader:
        glUseProgram(shaders[shaderIndex])
        currentShader = shaders[shaderIndex]
        changeShader = False
    
    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)
    glUniform3fv(
        glGetUniformLocation(currentShader, "color"),
        1,
        glm.value_ptr(color)
    )

    time = (pygame.time.get_ticks() - prev_time) / 1000
    #time = pygame.time.get_ticks()
    glUniform1f(
        glGetUniformLocation(currentShader, "time"),
        time
    )

    glDrawElements(GL_TRIANGLES, len(faces_data), GL_UNSIGNED_INT, None)

    calculateMatrix(r, vector)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                shaderIndex = 0
                changeShader = True
            if event.key == pygame.K_2:
                shaderIndex = 1
                changeShader = True
            if event.key == pygame.K_3:
                shaderIndex = 2
                changeShader = True
            if event.key == pygame.K_4:
                shaderIndex = 3
                changeShader = True

    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        vector = glm.vec3(0, 1, 0)
        r += 1
    if keys[pygame.K_LEFT]:
        vector = glm.vec3(0, 1, 0)
        r -= 1
    if keys[pygame.K_UP]:
        vector = glm.vec3(1, 0, 0)
        r += 1
    if keys[pygame.K_DOWN]:
        vector = glm.vec3(1, 0, 0)
        r -= 1
            
