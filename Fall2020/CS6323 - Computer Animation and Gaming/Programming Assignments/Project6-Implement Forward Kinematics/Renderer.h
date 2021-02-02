#pragma once

#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <nanogui/nanogui.h>

#include "Shader.h"
#include "Camera.h"
#include "Object.h"
#include "Lighting.h"
#include "Bone_Animation.h"

class Renderer
{
public:
	GLFWwindow* m_window;

	static Camera* m_camera;

	static Lighting* m_lightings;

	static nanogui::Screen* m_nanogui_screen;

	static Bone_Animation* m_bone_animation;

	std::vector<Object> obj_list;

	glm::vec4 background_color = glm::vec4(0.1,0.1,0.1,0.1);

	bool is_scene_reset = true;

	std::string model_name;

	float delta_time = 0.0;
	float last_frame = 0.0;

	static bool keys[1024];

public:
	Renderer();
	~Renderer();

	void nanogui_init(GLFWwindow* window);
	void init();

	void display(GLFWwindow* window);
	void run();

	void camera_move();

	void load_models();

	void draw_scene(Shader& shader);
	void draw_object(Shader& shader, Object& object);

	void draw_axis(Shader& shader, const glm::mat4 axis_obj_mat);
	void draw_plane(Shader& shader);
	void draw_bones(Shader& shader, Bone_Animation* m_bone_animation);

	void bind_vaovbo(Object &cur_obj);

	void setup_uniform_values(Shader& shader);
	void scene_reset();
};

