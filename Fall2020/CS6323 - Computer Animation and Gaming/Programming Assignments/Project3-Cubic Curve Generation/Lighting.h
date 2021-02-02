#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Lighting {

	struct Direction_Light {

		bool status;
		glm::vec3 direction;

		glm::vec4 ambient;
		glm::vec4 diffuse;
		glm::vec4 specular;

	};


	struct Point_Light {

		bool status;
		glm::vec3 position;
		float constant;
		float linear;
		float quadratic;

		glm::vec4 ambient;
		glm::vec4 diffuse;
		glm::vec4 specular;

	};

public:

	Direction_Light direction_light;
	Point_Light point_light;

	Lighting() {

	}

	~Lighting() {}

	void init() 
	{
		direction_light.status = true;
		direction_light.direction = glm::vec3(-1.0f, -1.0f, -1.0f);
		direction_light.ambient = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
		direction_light.diffuse = glm::vec4(0.6f, 0.6f, 0.6f, 1.0f);
		direction_light.specular = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		point_light.status = true;
		point_light.position = glm::vec3(1.2f, 1.0f, 2.0f);
		point_light.ambient = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
		point_light.diffuse = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		point_light.specular = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		point_light.constant = 1.0f;
		point_light.linear = 0.09f;
		point_light.quadratic = 0.032f;
	};
};