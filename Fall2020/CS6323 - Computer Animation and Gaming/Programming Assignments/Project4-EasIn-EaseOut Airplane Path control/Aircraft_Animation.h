#pragma once

#include <vector>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Curve.h"

class Aircraft_Animation
{

public:
	float total_moving_time = 10;
	float t1 = 0.1;
	float t2 = 0.7;

	float moving_time = 0.0f;
	float curve_length;
	float max_speed;

private:
	glm::mat4 m_model_mat;
	Curve* m_animation_curve = nullptr;

public:
	Aircraft_Animation();
	~Aircraft_Animation();

	void init();
	void init(Curve* animation_curve);

	void update(float delta_time);
	float ease(float curr_time);
	int linear_search(float distance);
	int binary_search(float distance);

	void reset();
	glm::mat4 get_model_mat() { return m_model_mat; };
};

