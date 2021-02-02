#version 330 core

in vec3 final_color;

layout( location = 0 ) out vec4 FragColor;

void main(){
    FragColor = vec4(final_color,1.0f);
}