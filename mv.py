import pygame
import glm
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import * 

pygame.init()

screen = pygame.display.set_mode(
    (1600, 1200),
    pygame.OPENGL | pygame.DOUBLEBUF
)

vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;

uniform mat4 matrix;

out vec3 ourColor;
void main()
{
    gl_Position = matrix * vec4(position, 1.0f);
    ourColor = vertexColor;
}
"""

fragment_shader = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;
in vec3 ourColor;

void main()
{
    //fragColor = vec4(ourColor, 1.0f);
    fragColor = vec4(color, 1.0f);

}
"""

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
shader = compileProgram(
    compiled_vertex_shader, 
    compiled_fragment_shader
)
glUseProgram(shader)

vertex_data = np.array([
    -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
     0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
     0.0,  0.5, 0.0, 0.0, 0.0, 1.0,
], dtype=np.float32)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)

glBufferData(
    GL_ARRAY_BUFFER, # tipo de datos
    vertex_data.nbytes, # tamaño de los datos en bytes
    vertex_data, # puntero a la data
    GL_STATIC_DRAW # tipo de uso de la data
)

vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

glVertexAttribPointer(
    0, 
    3,
    GL_FLOAT,
    GL_FALSE,
    6 * 4,
    ctypes.c_void_p(0)
)

glEnableVertexAttribArray(0)

glVertexAttribPointer(
    1, 
    3,
    GL_FLOAT,
    GL_FALSE,
    6 * 4,
    ctypes.c_void_p(3 * 4)
)

glEnableVertexAttribArray(1)

def calculateMatrix(angle):

    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), glm.vec3(0, 1, 0))
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

    glViewport(0, 0, 1600, 1200)

    matrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, "matrix"),
        1,
        GL_FALSE,
        glm.value_ptr(matrix)
    )

running = True

glClearColor(0.5, 1.0, 0.5, 1.0)
r = 0 

while running:
    r+=1
    glClear(GL_COLOR_BUFFER_BIT)
    

    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)

    glUniform3fv(
        glGetUniformLocation(shader, "color"),
        1,
        glm.value_ptr(color)
    )

    #pygame.time.wait(100)

    calculateMatrix(r)

    glDrawArrays(GL_TRIANGLES, 0, 3)
    
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
