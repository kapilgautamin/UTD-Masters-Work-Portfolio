#include "Renderer.h"

Camera* Renderer::m_camera = new Camera();

Lighting* Renderer::m_lightings = new Lighting();

Animation* Renderer::m_animation = new Animation();

nanogui::Screen* Renderer::m_nanogui_screen = nullptr;

bool Renderer::keys[1024];
float angleA = 20.0f;
float angleB = 60.0f;

Renderer::Renderer()
{
}


Renderer::~Renderer()
{	
}

void Renderer::nanogui_init(GLFWwindow* window)
{
	m_nanogui_screen = new nanogui::Screen();
	m_nanogui_screen->initialize(window, true);

	glViewport(0, 0, m_camera->width, m_camera->height);

	//glfwSwapInterval(0);
	//glfwSwapBuffers(window);

	// Create nanogui gui
	nanogui::FormHelper *gui_1 = new nanogui::FormHelper(m_nanogui_screen);
	nanogui::ref<nanogui::Window> nanoguiWindow_1 = gui_1->addWindow(Eigen::Vector2i(0, 0), "Nanogui control bar_1");

	//screen->setPosition(Eigen::Vector2i(-width/2 + 200, -height/2 + 300));

	gui_1->addGroup("Camera Position");
	gui_1->addVariable("X", m_camera->position[0])->setSpinnable(true);
	gui_1->addVariable("Y", m_camera->position[1])->setSpinnable(true);
	gui_1->addVariable("Z", m_camera->position[2])->setSpinnable(true);

	gui_1->addButton("Reset Camera", []() {
		m_camera->reset();
	});

	gui_1->addGroup("Camera Rotation");
	gui_1->addVariable("Rotate LCS by a", angleA)->setSpinnable(true);
	gui_1->addVariable("Rotate WCS by b", angleB)->setSpinnable(true);

	gui_1->addButton("Rotate Local X", []() {
		m_animation->rotateLCS(angleA);
	});

	gui_1->addButton("Rotate Global Y", []() {
		m_animation->rotateWCS(angleB);
	});

	gui_1->addButton("Reset model", []() {
		m_animation->reset();
	});

	m_nanogui_screen->setVisible(true);
	m_nanogui_screen->performLayout();

	glfwSetCursorPosCallback(window,
		[](GLFWwindow *window, double x, double y) {
		m_nanogui_screen->cursorPosCallbackEvent(x, y);
	}
	);

	glfwSetMouseButtonCallback(window,
		[](GLFWwindow *, int button, int action, int modifiers) {
		m_nanogui_screen->mouseButtonCallbackEvent(button, action, modifiers);
	}
	);

	glfwSetKeyCallback(window,
		[](GLFWwindow *window, int key, int scancode, int action, int mods) {
		//screen->keyCallbackEvent(key, scancode, action, mods);

		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);
		if (key >= 0 && key < 1024)
		{
			if (action == GLFW_PRESS)
				keys[key] = true;
			else if (action == GLFW_RELEASE)
				keys[key] = false;
		}
	}
	);

	glfwSetCharCallback(window,
		[](GLFWwindow *, unsigned int codepoint) {
		m_nanogui_screen->charCallbackEvent(codepoint);
	}
	);

	glfwSetDropCallback(window,
		[](GLFWwindow *, int count, const char **filenames) {
		m_nanogui_screen->dropCallbackEvent(count, filenames);
	}
	);

	glfwSetScrollCallback(window,
		[](GLFWwindow *, double x, double y) {
		m_nanogui_screen->scrollCallbackEvent(x, y);
		//m_camera->ProcessMouseScroll(y);
	}
	);

	glfwSetFramebufferSizeCallback(window,
		[](GLFWwindow *, int width, int height) {
		m_nanogui_screen->resizeCallbackEvent(width, height);
	}
	);
}

void Renderer::init()
{
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	m_camera->init();
	m_lightings->init();
	m_animation->init();

	// Create a GLFWwindow object that we can use for GLFW's functions
	this->m_window = glfwCreateWindow(m_camera->width, m_camera->height, "Assignment 1", nullptr, nullptr);
	glfwMakeContextCurrent(this->m_window);
	glfwSetWindowAspectRatio(this->m_window, 16, 9);

	glewExperimental = GL_TRUE;
	glewInit();

	nanogui_init(this->m_window);
}

void Renderer::display(GLFWwindow* window)
{
	Shader m_shader = Shader("./shader/basic.vert", "./shader/basic.frag");

	// Main frame while loop
	while (!glfwWindowShouldClose(window))
	{

		glfwPollEvents();

		if (is_scean_reset) {
			scean_reset();
			is_scean_reset = false;
		}

		camera_move();

		m_shader.use();
			
		setup_uniform_values(m_shader);

		draw_scene(m_shader);

		m_nanogui_screen->drawWidgets();

		// Swap the screen buffers
		glfwSwapBuffers(window);
	}

	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return;
}

void Renderer::run()
{
	init();
	display(this->m_window);
}

void Renderer::load_models()
{
	obj_list.clear();
	Object main_object("./objs/bunny.obj");
	main_object.obj_color = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);
	main_object.obj_name = "main_object";

	Object plane_object("./objs/plane.obj");
	plane_object.obj_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
	plane_object.obj_name = "plane";

	Object arrow_object("./objs/arrow.obj");
	arrow_object.obj_name = "axis_arrow";

	bind_vaovbo(main_object);
	bind_vaovbo(plane_object);
	bind_vaovbo(arrow_object);

	// Here we only load one model
	obj_list.push_back(main_object);
	obj_list.push_back(plane_object);
	obj_list.push_back(arrow_object);
}

void Renderer::draw_scene(Shader& shader)
{
	// Set up some basic parameters
	glClearColor(background_color[0], background_color[1], background_color[2], background_color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);

	glFrontFace(GL_CW);

	for (size_t i = 0; i < obj_list.size(); i++)
	{

		if (obj_list[i].obj_name == "main_object")
		{
			// Before draw the model, change its model mat
			glm::mat4 main_object_model_mat =  glm::mat4(1.0f);

			m_animation->update(delta_time);

			main_object_model_mat = m_animation->get_model_mat();

			glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(main_object_model_mat));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));

			draw_object(shader,obj_list[i]);
		}

		if (obj_list[i].obj_name == "plane")
		{
			// Draw plane
			glm::mat4 plane_model_mat =  glm::mat4(1.0f);
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(plane_model_mat));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));
			draw_object(shader, obj_list[i]);
		}

		if (obj_list[i].obj_name == "axis_arrow")
		{
			// Draw three axis
			glm::mat4 model_mat_x = glm::mat4(1.0f);
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(model_mat_x));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));
			obj_list[i].obj_color = glm::vec4(1, 0, 0, 1);
			draw_object(shader, obj_list[i]);

			glm::mat4 model_mat_y = glm::mat4(1.0f);
			model_mat_y = glm::rotate(model_mat_y, glm::radians(-90.0f), glm::vec3(0.0f,1.0f,0.0f));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(model_mat_y));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));
			obj_list[i].obj_color = glm::vec4(0, 1, 0, 1);
			draw_object(shader,obj_list[i]);

			glm::mat4 model_mat_z = glm::mat4(1.0f);
			model_mat_z = glm::rotate(model_mat_z, glm::radians(90.0f), glm::vec3(0.0f,0.0f,1.0f));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(model_mat_z));
			glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));
			obj_list[i].obj_color = glm::vec4(0, 0, 1, 1);
			draw_object(shader,obj_list[i]);
		}
	}
}

void Renderer::camera_move()
{
	GLfloat current_frame = glfwGetTime();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	// Camera controls
	if (keys[GLFW_KEY_W])
		m_camera->process_keyboard(FORWARD, delta_time);
	if (keys[GLFW_KEY_S])
		m_camera->process_keyboard(BACKWARD, delta_time);
	if (keys[GLFW_KEY_A])
		m_camera->process_keyboard(LEFT, delta_time);
	if (keys[GLFW_KEY_D])
		m_camera->process_keyboard(RIGHT, delta_time);
	if (keys[GLFW_KEY_Q])
		m_camera->process_keyboard(UP, delta_time);
	if (keys[GLFW_KEY_E])
		m_camera->process_keyboard(DOWN, delta_time);
	if (keys[GLFW_KEY_I])
		m_camera->process_keyboard(ROTATE_X_UP, delta_time);
	if (keys[GLFW_KEY_K])
		m_camera->process_keyboard(ROTATE_X_DOWN, delta_time);
	if (keys[GLFW_KEY_J])
		m_camera->process_keyboard(ROTATE_Y_UP, delta_time);
	if (keys[GLFW_KEY_L])
		m_camera->process_keyboard(ROTATE_Y_DOWN, delta_time);
	if (keys[GLFW_KEY_U])
		m_camera->process_keyboard(ROTATE_Z_UP, delta_time);
	if (keys[GLFW_KEY_O])
		m_camera->process_keyboard(ROTATE_Z_DOWN, delta_time);
}

void Renderer::draw_object(Shader& shader, Object& object)
{
	glBindVertexArray(object.vao);

	glUniform3f(glGetUniformLocation(shader.program, "m_object.object_color"), object.obj_color[0], object.obj_color[1], object.obj_color[2]);
	glUniform1f(glGetUniformLocation(shader.program, "m_object.shininess"), object.shininess);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawArrays(GL_TRIANGLES, 0, object.vao_vertices.size());
	glBindVertexArray(0);
}

void Renderer::bind_vaovbo(Object &cur_obj)
{
	glGenVertexArrays(1, &cur_obj.vao);
	glGenBuffers(1, &cur_obj.vbo);

	glBindVertexArray(cur_obj.vao);

	glBindBuffer(GL_ARRAY_BUFFER, cur_obj.vbo);
	glBufferData(GL_ARRAY_BUFFER, cur_obj.vao_vertices.size() * sizeof(Object::Vertex), &cur_obj.vao_vertices[0], GL_STATIC_DRAW);

	// Vertex Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Object::Vertex), (GLvoid*)0);
	// Vertex Normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Object::Vertex), (GLvoid*)offsetof(Object::Vertex, Normal));
	// Vertex Texture Coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Object::Vertex), (GLvoid*)offsetof(Object::Vertex, TexCoords));

	glBindVertexArray(0);
}

void Renderer::setup_uniform_values(Shader& shader)
{
	// Camera uniform values
	glUniform3f(glGetUniformLocation(shader.program, "camera_pos"), m_camera->position.x, m_camera->position.y, m_camera->position.z);

	glUniformMatrix4fv(glGetUniformLocation(shader.program, "projection"), 1, GL_FALSE, glm::value_ptr(m_camera->get_projection_mat()));

	// Light uniform values
	glUniform1i(glGetUniformLocation(shader.program, "dir_light.status"), m_lightings->direction_light.status);
	glUniform3f(glGetUniformLocation(shader.program, "dir_light.direction"), m_lightings->direction_light.direction[0], m_lightings->direction_light.direction[1], m_lightings->direction_light.direction[2]);
	glUniform3f(glGetUniformLocation(shader.program, "dir_light.ambient"), m_lightings->direction_light.ambient[0], m_lightings->direction_light.ambient[1], m_lightings->direction_light.ambient[2]);
	glUniform3f(glGetUniformLocation(shader.program, "dir_light.diffuse"), m_lightings->direction_light.diffuse[0], m_lightings->direction_light.diffuse[1], m_lightings->direction_light.diffuse[2]);
	glUniform3f(glGetUniformLocation(shader.program, "dir_light.specular"), m_lightings->direction_light.specular[0], m_lightings->direction_light.specular[1], m_lightings->direction_light.specular[2]);

	// Set current point light as camera's position
	m_lightings->point_light.position = m_camera->position;
	glUniform1i(glGetUniformLocation(shader.program, "point_light.status"), m_lightings->point_light.status);
	glUniform3f(glGetUniformLocation(shader.program, "point_light.position"), m_lightings->point_light.position[0], m_lightings->point_light.position[1], m_lightings->point_light.position[2]);
	glUniform3f(glGetUniformLocation(shader.program, "point_light.ambient"), m_lightings->point_light.ambient[0], m_lightings->point_light.ambient[1], m_lightings->point_light.ambient[2]);
	glUniform3f(glGetUniformLocation(shader.program, "point_light.diffuse"), m_lightings->point_light.diffuse[0], m_lightings->point_light.diffuse[1], m_lightings->point_light.diffuse[2]);
	glUniform3f(glGetUniformLocation(shader.program, "point_light.specular"), m_lightings->point_light.specular[0], m_lightings->point_light.specular[1], m_lightings->point_light.specular[2]);
	glUniform1f(glGetUniformLocation(shader.program, "point_light.constant"), m_lightings->point_light.constant);
	glUniform1f(glGetUniformLocation(shader.program, "point_light.linear"), m_lightings->point_light.linear);
	glUniform1f(glGetUniformLocation(shader.program, "point_light.quadratic"), m_lightings->point_light.quadratic);
}

void Renderer::scean_reset()
{
	load_models();
	m_camera->reset();
}