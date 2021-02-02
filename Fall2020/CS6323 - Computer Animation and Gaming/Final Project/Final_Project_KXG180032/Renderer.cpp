#include "Renderer.h"
#include <GL\wglew.h>
#include <GL\freeglut_std.h>

Camera* Renderer::m_camera = new Camera();

Lighting* Renderer::m_lightings = new Lighting();

Cloth_Animation* Renderer::m_cloth_animation = new Cloth_Animation();
nanogui::Screen* Renderer::m_nanogui_screen = nullptr;

bool Renderer::keys[1024];
bool is_left_mouse_button_clicked = false;
bool mouse_offset = false;
double lastX, lastY;

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
	nanogui::FormHelper* gui_1 = new nanogui::FormHelper(m_nanogui_screen);
	nanogui::ref<nanogui::Window> nanoguiWindow_1 = gui_1->addWindow(Eigen::Vector2i(0, 0), "Nanogui control bar_1");

	//screen->setPosition(Eigen::Vector2i(-width/2 + 200, -height/2 + 300));

	gui_1->addGroup("Camera Position");
	static auto camera_x_widget = gui_1->addVariable("X", m_camera->position[0]);
	static auto camera_y_widget = gui_1->addVariable("Y", m_camera->position[1]);
	static auto camera_z_widget = gui_1->addVariable("Z", m_camera->position[2]);

	gui_1->addButton("Reset Camera", []() {
		m_camera->reset();
		camera_x_widget->setValue(m_camera->position[0]);
		camera_y_widget->setValue(m_camera->position[1]);
		camera_z_widget->setValue(m_camera->position[2]);
		});

	gui_1->addGroup("Cloth configuration");

	display_mass = gui_1->addVariable("Mass", m_cloth_animation->mass);
	windPower = gui_1->addVariable("Wind Power", m_cloth_animation->windStrength);
	gravityPower = gui_1->addVariable("Gravity Power", m_cloth_animation->gravityStrength);

	gravity = gui_1->addVariable("Enable gravity", m_cloth_animation->enable_gravity);
	opposite_gravity = gui_1->addVariable("Anti gravity", m_cloth_animation->anti_gravity);
	force_spring = gui_1->addVariable("Spring forces", m_cloth_animation->spring_forces);
	windy = gui_1->addVariable("Enable wind", m_cloth_animation->enable_wind);

	points = gui_1->addVariable("Show points", m_cloth_animation->show_points);
	fill = gui_1->addVariable("Fill cloth", m_cloth_animation->fill_cloth);

	gui_1->addGroup("Release corners");
	release1 = gui_1->addVariable("Release corner 1", m_cloth_animation->release_corner_1);
	release2 = gui_1->addVariable("Release corner 2", m_cloth_animation->release_corner_2);
	release3 = gui_1->addVariable("Release corner 3", m_cloth_animation->release_corner_3);
	release4 = gui_1->addVariable("Release corner 4", m_cloth_animation->release_corner_4);

	display_mass->setSpinnable(true);
	windPower->setSpinnable(true);
	gravityPower->setSpinnable(true);

	display_mass->setFixedWidth(80);

	gui_1->addButton("Reset Cloth", [&]() {
		m_cloth_animation->reset();

		Renderer::load_cloth();
	});

	sphere_x = gui_1->addVariable("Sphere X", m_cloth_animation->sphere_position[0]);
	sphere_y = gui_1->addVariable("Sphere Y", m_cloth_animation->sphere_position[1]);
	sphere_z = gui_1->addVariable("Sphere Z", m_cloth_animation->sphere_position[2]);

	wind_x = gui_1->addVariable("Wind x", m_cloth_animation->wind_position[0]);
	wind_y = gui_1->addVariable("Wind y", m_cloth_animation->wind_position[1]);
	wind_z = gui_1->addVariable("Wind z", m_cloth_animation->wind_position[2]);

	m_nanogui_screen->setVisible(true);
	m_nanogui_screen->performLayout();

	glfwSetCursorPosCallback(window,
		[](GLFWwindow* window, double x, double y) {
			m_nanogui_screen->cursorPosCallbackEvent(x, y);

			if (is_left_mouse_button_clicked) {
				if (mouse_offset) {
					lastX = x;
					lastY = y;
					mouse_offset = false;
				}else {
					float delta_interval = 0.006f;
					if (x > lastX) {
						//std::cout << "RIGHT" << std::endl;
						m_camera->process_keyboard(ROTATE_Y_DOWN, delta_interval);
					}
					else {
						//std::cout << "LEFT" << std::endl;
						m_camera->process_keyboard(ROTATE_Y_UP, delta_interval);
					}

					if (y > lastY) {
						//std::cout << "DOWN" << std::endl;
						m_camera->process_keyboard(ROTATE_X_DOWN, delta_interval);

					}
					else {
						//std::cout << "UP" << std::endl;
						m_camera->process_keyboard(ROTATE_X_UP, delta_interval);

					}
					lastX = x;
					lastY = y;
				}
			}
		}
	);

	glfwSetMouseButtonCallback(window,
		[](GLFWwindow*, int button, int action, int modifiers) {
			m_nanogui_screen->mouseButtonCallbackEvent(button, action, modifiers);
			//std::cout << "button:" << button << std::endl;
			//std::cout << "action:" << action << std::endl;
			//std::cout << "modifiers:" << modifiers << std::endl;
			if (button == 0) {	//left mouse button
				if (action == 1) {	//clicked
					is_left_mouse_button_clicked = true;
					mouse_offset = true;
				}
				else {
					is_left_mouse_button_clicked = false;
					mouse_offset = false;
				}
			}
		}
	);

	glfwSetKeyCallback(window,
		[](GLFWwindow* window, int key, int scancode, int action, int mods) {
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
			camera_x_widget->setValue(m_camera->position[0]);
			camera_y_widget->setValue(m_camera->position[1]);
			camera_z_widget->setValue(m_camera->position[2]);
		}
	);

	glfwSetCharCallback(window,
		[](GLFWwindow*, unsigned int codepoint) {
			m_nanogui_screen->charCallbackEvent(codepoint);
		}
	);

	glfwSetDropCallback(window,
		[](GLFWwindow*, int count, const char** filenames) {
			m_nanogui_screen->dropCallbackEvent(count, filenames);
		}
	);

	glfwSetScrollCallback(window,
		[](GLFWwindow*, double x, double y) {
			m_nanogui_screen->scrollCallbackEvent(x, y);
			m_camera->process_mouse_scroll(y * 0.04f);
		}
	);

	glfwSetFramebufferSizeCallback(window,
		[](GLFWwindow*, int width, int height) {
			m_nanogui_screen->resizeCallbackEvent(width, height);
		}
	);

}

void Renderer::load_cloth() {
	
	Object cloth_object(m_cloth_animation->position_vec);
	cloth_object.obj_color = glm::vec4(1.0, 0.0, 0.0, 1.0);
	cloth_object.m_render_type = RENDER_POINTS;
	//cloth_object.m_obj_type = OBJ_POINTS;
	cloth_object.obj_name = "cloth";

	bind_vaovbo(cloth_object);
	obj_list.push_back(cloth_object);
}

void Renderer::load_models()
{
	obj_list.clear();

	Object plane_object("./objs/plane.obj");
	plane_object.obj_color = glm::vec4(0.5, 0.5, 0.5, 1.0);
	plane_object.obj_name = "plane";

	Object arrow_object("./objs/arrow.obj");
	arrow_object.obj_name = "axis_arrow";

	Object sphere_object("./objs/sphere.obj");
	sphere_object.obj_color = glm::vec4(1.0, 1.0, 0.0, 1.0);
	sphere_object.obj_name = "sphere";

	bind_vaovbo(plane_object);
	bind_vaovbo(arrow_object);
	bind_vaovbo(sphere_object);

	obj_list.push_back(plane_object);
	obj_list.push_back(arrow_object);
	obj_list.push_back(sphere_object);

	load_cloth();
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

	glm::mat4 world_identity_obj_mat = glm::mat4(1.0f);
	draw_axis(shader, world_identity_obj_mat);
	draw_cloth(shader);
	draw_wind(shader);
	draw_plane(shader);
	draw_sphere(shader);
	animate_cloth(shader, m_cloth_animation);
}

void Renderer::camera_move()
{
	float current_frame = glfwGetTime();
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

	if (object.m_render_type == RENDER_TRIANGLES)
	{
		if (object.m_obj_type == OBJ_POINTS)
		{
			std::cout << "Error: Cannot render triangles if input obj type is point\n";
			return;
		}
		if (object.m_obj_type == OBJ_TRIANGLES)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawArrays(GL_TRIANGLES, 0, object.vao_vertices.size());
		}
	}

	if (object.m_render_type == RENDER_LINES)
	{
		glLineWidth(10.0);
		if (object.m_obj_type == OBJ_POINTS)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawArrays(GL_LINE_LOOP, 0, object.vao_vertices.size());
		}
		if (object.m_obj_type == OBJ_TRIANGLES)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawArrays(GL_TRIANGLES, 0, object.vao_vertices.size());
		}
	}

	if (object.m_obj_type == OBJ_POINTS)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINTS);
		glDrawArrays(GL_POINTS, 0, object.vao_vertices.size());
	}
	glBindVertexArray(0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Renderer::draw_axis(Shader& shader, const glm::mat4 axis_obj_mat)
{
	// You can always see the arrow
	glDepthFunc(GL_ALWAYS);
	// Get arrow obj
	Object* arrow_obj = nullptr;
	for (unsigned int i = 0; i < obj_list.size(); i++)
	{
		if (obj_list[i].obj_name == "axis_arrow") {
			arrow_obj = &obj_list[i];
		}
	}

	if (arrow_obj == nullptr)
		return;

	// Draw main axis
	arrow_obj->obj_mat = axis_obj_mat;
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(arrow_obj->obj_mat));
	arrow_obj->obj_color = glm::vec4(1, 0, 0, 1);
	draw_object(shader, *arrow_obj);

	arrow_obj->obj_mat = axis_obj_mat;
	arrow_obj->obj_mat = glm::rotate(arrow_obj->obj_mat, glm::radians(90.0f), glm::vec3(0, 0, 1));
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(arrow_obj->obj_mat));
	arrow_obj->obj_color = glm::vec4(0, 1, 0, 1);
	draw_object(shader, *arrow_obj);

	arrow_obj->obj_mat = axis_obj_mat;
	arrow_obj->obj_mat = glm::rotate(arrow_obj->obj_mat, glm::radians(-90.0f), glm::vec3(0, 1, 0));
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(arrow_obj->obj_mat));
	arrow_obj->obj_color = glm::vec4(0, 0, 1, 1);
	draw_object(shader, *arrow_obj);
	glDepthFunc(GL_LESS);
}

void Renderer::draw_plane(Shader& shader)
{
	Object* plane_obj = nullptr;
	for (unsigned int i = 0; i < obj_list.size(); i++)
	{
		if (obj_list[i].obj_name == "plane") {
			plane_obj = &obj_list[i];
		}
	}
	if (plane_obj == nullptr)
		return;

	plane_obj->obj_mat = glm::mat4(1.0f);
	plane_obj->obj_mat = glm::scale(plane_obj->obj_mat, glm::vec3(10, 10, 10));
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(plane_obj->obj_mat));
	draw_object(shader, *plane_obj);
}

void Renderer::draw_cloth(Shader& shader) {

	Object* cloth = nullptr;
	for (unsigned int i = 0; i < obj_list.size(); i++)
	{
		if (obj_list[i].obj_name == "cloth") {
			cloth = &obj_list[i];
		}
	}

	if (cloth == nullptr)
		return;

	//draw_object(shader, *cloth);

	m_cloth_animation->triangles.clear();
	m_cloth_animation->points.clear();

	//draw polygons
	for (int i = 0; i < m_cloth_animation->indices.size(); i += 3) {
		m_cloth_animation->triangles.push_back(m_cloth_animation->position_vec[m_cloth_animation->indices[i]]);
		m_cloth_animation->triangles.push_back(m_cloth_animation->position_vec[m_cloth_animation->indices[i + 1]]);
		m_cloth_animation->triangles.push_back(m_cloth_animation->position_vec[m_cloth_animation->indices[i + 2]]);
	}

	Object cloth_triangles(m_cloth_animation->triangles);
	cloth_triangles.obj_color = glm::vec4(0.0f, 0.0f, 0.7f, 1.0);
	
	cloth_triangles.m_render_type = RENDER_LINES;
	if(m_cloth_animation->fill_cloth)
		cloth_triangles.m_render_type = RENDER_TRIANGLES;

	cloth_triangles.m_obj_type = OBJ_TRIANGLES;
	bind_vaovbo(cloth_triangles);
	cloth_triangles.obj_mat = glm::mat4(1.0f);
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(cloth_triangles.obj_mat));
	
	draw_object(shader, cloth_triangles);

	if (m_cloth_animation->show_points) {
		int total_points = (m_cloth_animation->numX + 1) * (m_cloth_animation->numY + 1);
		for (int i = 0; i < total_points; i++) {
			glm::vec3 p = m_cloth_animation->position_vec[i];
			m_cloth_animation->points.push_back(p);
		}

		Object cloth_points(m_cloth_animation->points);
		cloth_points.obj_color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0);
		cloth_points.m_render_type = RENDER_POINTS;
		cloth_points.m_obj_type = OBJ_POINTS;
		bind_vaovbo(cloth_points);
		glPointSize(5);
		cloth_points.obj_mat = glm::mat4(1.0f);
		glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(cloth_points.obj_mat));
		draw_object(shader, cloth_points);
	}
}

void Renderer::draw_sphere(Shader& shader)
{
	Object* sphere_obj = nullptr;
	for (unsigned int i = 0; i < obj_list.size(); i++)
	{
		if (obj_list[i].obj_name == "sphere") {
			sphere_obj = &obj_list[i];
		}
	}
	if (sphere_obj == nullptr)
		return;

	sphere_obj->obj_mat = glm::mat4(1.0f);
	sphere_obj->obj_mat = glm::translate(sphere_obj->obj_mat, m_cloth_animation->sphere_position);
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(sphere_obj->obj_mat));
	sphere_obj->obj_color = m_cloth_animation->colors[3];
	draw_object(shader, *sphere_obj);
}

void Renderer::draw_wind(Shader& shader)
{
	glm::vec3 dir = glm::normalize(m_cloth_animation->wind * m_cloth_animation->windStrength);
	std::vector<glm::vec3> wind_vec;
	wind_vec.push_back(dir);

	Object wind_object(wind_vec);
	wind_object.obj_color = glm::vec4(1.0, 1.0, 1.0, 1.0);
	wind_object.m_render_type = RENDER_LINES;
	//wind_object.m_obj_type = OBJ_POINTS;
	wind_object.obj_name = "wind";

	bind_vaovbo(wind_object);

	wind_object.obj_mat = glm::mat4(1.0f);
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm::value_ptr(wind_object.obj_mat));
	draw_object(shader, wind_object);
}

void Renderer::animate_cloth(Shader& shader, Cloth_Animation* m_cloth_animation)
{
	float rx = (float(rand()) / RAND_MAX) * 2 - 1.0f;
	float ry = (float(rand()) / RAND_MAX) * 2 - 1.0f;
	float rz = (float(rand()) / RAND_MAX) * 2 - 1.0f;
	m_cloth_animation->wind += glm::vec3(rx, ry, rz) * 0.00001f;
	
	Object* sphere = nullptr;
	for (unsigned int i = 0; i < obj_list.size(); i++)
	{
		if (obj_list[i].obj_name == "sphere") {
			sphere = &obj_list[i];
		}
	}
	if (sphere == nullptr)
		return;

	m_cloth_animation->sphere_center = sphere->obj_center;
	m_cloth_animation->sphere = sphere->obj_mat;
	m_cloth_animation->inverse_sphere = glm::inverse(m_cloth_animation->sphere);

	m_cloth_animation->update(delta_time);
}

void Renderer::bind_vaovbo(Object& cur_obj)
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
	glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm::value_ptr(m_camera->get_view_mat()));

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

void Renderer::init()
{
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

#if defined(__APPLE__)
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	m_camera->init();

	// Create a GLFWwindow object that we can use for GLFW's functions
	this->m_window = glfwCreateWindow(m_camera->width, m_camera->height, "Spring-damper cloth", nullptr, nullptr);
	glfwMakeContextCurrent(this->m_window);

	glewExperimental = GL_TRUE;
	glewInit();

	m_lightings->init();
	m_cloth_animation->init_cloth();
	m_cloth_animation->init();
	nanogui_init(this->m_window);
}

void Renderer::display(GLFWwindow* window)
{
	Shader m_shader = Shader("./shader/basic.vert", "./shader/basic.frag");

	// Main frame while loop
	while (!glfwWindowShouldClose(window))
	{

		glfwPollEvents();

		if (is_scene_reset) {
			scene_reset();
			is_scene_reset = false;
		}

		camera_move();

		display_mass->setValue(m_cloth_animation->mass);
		windPower->setValue(m_cloth_animation->windStrength);
		gravityPower->setValue(m_cloth_animation->gravityStrength);

		gravity->setValue(m_cloth_animation->enable_gravity);
		opposite_gravity->setValue(m_cloth_animation->anti_gravity);
		force_spring->setValue(m_cloth_animation->spring_forces);
		windy->setValue(m_cloth_animation->enable_wind);

		release1->setValue(m_cloth_animation->release_corner_1);
		release2->setValue(m_cloth_animation->release_corner_2);
		release3->setValue(m_cloth_animation->release_corner_3);
		release4->setValue(m_cloth_animation->release_corner_4);

		points->setValue(m_cloth_animation->show_points);
		fill->setValue(m_cloth_animation->fill_cloth);

		sphere_x->setValue(m_cloth_animation->sphere_position[0]);
		sphere_y->setValue(m_cloth_animation->sphere_position[1]);
		sphere_z->setValue(m_cloth_animation->sphere_position[2]);

		wind_x->setValue(m_cloth_animation->wind_position[0]);
		wind_y->setValue(m_cloth_animation->wind_position[1]);
		wind_z->setValue(m_cloth_animation->wind_position[2]);

		m_shader.use();

		setup_uniform_values(m_shader);

		draw_scene(m_shader);

		m_nanogui_screen->drawWidgets();

		// Swap the screen buffers
		glfwSwapBuffers(window);
	}

	m_cloth_animation->position_vec.clear();
	m_cloth_animation->velocity_vec.clear();
	m_cloth_animation->force_vec.clear();
	m_cloth_animation->triangles.clear();
	m_cloth_animation->points.clear();
	
	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();

	return;
}

void Renderer::scene_reset()
{
	load_models();
	m_camera->reset();
	m_cloth_animation->reset();
}

void Renderer::run()
{
	init();
	display(this->m_window);
}