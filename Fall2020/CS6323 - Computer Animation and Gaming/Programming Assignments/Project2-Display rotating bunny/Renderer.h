#pragma once

#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <nanogui/nanogui.h>

#include "Shader.h"
#include "Camera.h"
#include "Object.h"
#include "Lighting.h"
#include "Animation.h"

class Renderer
{
public:
	GLFWwindow* m_window;

	static Camera* m_camera;

	static Lighting* m_lightings;

	static Animation* m_animation;

	static nanogui::Screen* m_nanogui_screen;

	std::vector<Object> obj_list;

	glm::vec4 background_color = glm::vec4(0.1f,0.1f,0.1f,0.1f);

	bool is_scean_reset = true;

	std::string model_name;

	GLfloat delta_time = 0.0f;
	GLfloat last_frame = 0.0f;

	static bool keys[1024];

public:
	Renderer();
	~Renderer();

	void nanogui_init(GLFWwindow* window);
	void init();

	void display(GLFWwindow* window);
	void run();

	void load_models();
	void draw_scene(Shader& shader);

	void camera_move();

	void draw_object(Shader& shader, Object& object);

	void bind_vaovbo(Object &cur_obj);

	void setup_uniform_values(Shader& shader);
	void scean_reset();
};

