#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>	

enum OBJ_TYPE
{
	OBJ_POINTS,
	OBJ_TRIANGLES
};

enum RENDER_TYPE
{
	RENDER_POINTS,
	RENDER_LINES,
	RENDER_TRIANGLES
};

class Object
{

public:
	struct Vertex {
		// Position
		glm::vec3 Position;
		// Normal
		glm::vec3 Normal;
		// TexCoords
		glm::vec2 TexCoords;
	};

	struct Vertex_Index {
		int pos_idx;
		int normal_idx;
		int texcoord_idx;
	};

	struct Face_Index {
		Vertex_Index vertex[3];
	};

	// veo and vao vector
	std::vector<Vertex> vao_vertices;
	std::vector<unsigned int> veo_indices;

	// obj original data vector
	std::vector<glm::vec3> ori_positions;
	std::vector<glm::vec3> ori_normals;
	std::vector<glm::vec2> ori_texcoords;

	// obj face index vector
	std::vector<Face_Index> indexed_faces;

	glm::vec3 obj_center;
	
	glm::vec4 obj_color = glm::vec4(0.7, 0.7, 0.7, 1.0);
	GLfloat shininess = 32.0f;

	std::string m_obj_path;
	std::string obj_name;

	GLuint vao, vbo;

	OBJ_TYPE m_obj_type;
	RENDER_TYPE m_render_type;

public:
	void load_obj(std::string obj_path)
	{
		int path_str_length = obj_path.size();
		std::string suffix = obj_path.substr(path_str_length - 3, path_str_length);

		if (suffix == "obj") {
			this->vao_vertices.clear();
			this->veo_indices.clear();
			this->indexed_faces.clear();

			this->ori_positions.clear();
			this->ori_normals.clear();
			this->ori_texcoords.clear();

			std::ifstream ifs;
			// Store original data vector
			try {
				ifs.open(obj_path);
				std::string one_line;
				while (getline(ifs, one_line)) {
					std::stringstream ss(one_line);
					std::string type;
					ss >> type;
					if (type == "v") {
						glm::vec3 vert_pos;
						ss >> vert_pos[0] >> vert_pos[1] >> vert_pos[2];
						this->ori_positions.push_back(vert_pos);
					}
					else if (type == "vt") {
						glm::vec2 tex_coord;
						ss >> tex_coord[0] >> tex_coord[1];
						this->ori_texcoords.push_back(tex_coord);
					}
					else if (type == "vn") {
						glm::vec3 vert_norm;
						ss >> vert_norm[0] >> vert_norm[1] >> vert_norm[2];
						this->ori_normals.push_back(vert_norm);
					}
					else if (type == "f") {
						Face_Index face_idx;
						// Here we only accept face number 3
						for (int i = 0; i < 3; i++) {
							std::string s_vertex;
							ss >> s_vertex;
							int pos_idx = -1;
							int tex_idx = -1;
							int norm_idx = -1;
							sscanf(s_vertex.c_str(), "%d/%d/%d", &pos_idx, &tex_idx, &norm_idx);
							// We have to use index -1 because the obj index starts at 1
							// Incorrect input will be set as -1
							face_idx.vertex[i].pos_idx = pos_idx > 0 ? pos_idx - 1 : -1;
							face_idx.vertex[i].texcoord_idx = tex_idx > 0 ? tex_idx - 1 : -1;
							face_idx.vertex[i].normal_idx = norm_idx > 0 ? norm_idx - 1 : -1;
						}
						indexed_faces.push_back(face_idx);
					}
				}
			}
			catch (const std::exception&) {
				std::cout << "Error: Obj file cannot be read\n";
			}

			// Retrieve data from index and assign to vao and veo
			for (int i = 0; i < indexed_faces.size(); i++) {
				Face_Index cur_idx_face = indexed_faces[i];
				// If no normal: recalculate for them
				glm::vec3 v0 = ori_positions[cur_idx_face.vertex[0].pos_idx];
				glm::vec3 v1 = ori_positions[cur_idx_face.vertex[1].pos_idx];
				glm::vec3 v2 = ori_positions[cur_idx_face.vertex[2].pos_idx];
				glm::vec3 new_normal = glm::cross(v1 - v0, v2 - v0);

				for (int j = 0; j < 3; j++) {
					Vertex cur_vertex;
					Vertex_Index cur_idx_vertex = cur_idx_face.vertex[j];
					if (cur_idx_vertex.pos_idx >= 0) {
						cur_vertex.Position = ori_positions[cur_idx_vertex.pos_idx];
					}
					if (cur_idx_vertex.normal_idx >= 0) {
						cur_vertex.Normal = ori_normals[cur_idx_vertex.normal_idx];
					}
					else {
						cur_vertex.Normal = new_normal;
					}
					if (cur_idx_vertex.texcoord_idx >= 0) {
						cur_vertex.TexCoords = ori_texcoords[cur_idx_vertex.texcoord_idx];
					}
					vao_vertices.push_back(cur_vertex);
					veo_indices.push_back(i * 3 + j);
				}
			}
		}
	};

	void calculate_center()
	{
		glm::vec3 max_bound(INT_MIN);
		glm::vec3 min_bound(INT_MAX);
		for (auto vertex : this->vao_vertices) {
			max_bound[0] = std::max(vertex.Position[0], max_bound[0]);
			max_bound[1] = std::max(vertex.Position[1], max_bound[1]);
			max_bound[2] = std::max(vertex.Position[2], max_bound[2]);
			min_bound[0] = std::min(vertex.Position[0], min_bound[0]);
			min_bound[1] = std::min(vertex.Position[1], min_bound[1]);
			min_bound[2] = std::min(vertex.Position[2], min_bound[2]);
		}
		this->obj_center = (max_bound + min_bound) * 0.5f;

	};

	// Initialize object using obj file
	Object(std::string obj_path) { 
		this->m_obj_path = obj_path; 
		load_obj(this->m_obj_path);
		calculate_center();
		m_obj_type = OBJ_TRIANGLES;
		m_render_type = RENDER_TRIANGLES;
	};
	
	// Initialize object using point array, can used to draw 
	Object(std::vector<glm::vec3> points)
	{
		this->vao_vertices.clear();
		this->veo_indices.clear();
		this->indexed_faces.clear();

		this->ori_positions.clear();
		this->ori_normals.clear();
		this->ori_texcoords.clear();
		for (unsigned int i = 0; i < points.size(); i++)
		{
			glm::vec3 cur_point = points[i];
			this->ori_positions.push_back(cur_point);
			Vertex cur_vertex;
			cur_vertex.Position = cur_point;
			this->vao_vertices.push_back(cur_vertex);
		}
		calculate_center();
		m_obj_type = OBJ_POINTS;
		m_render_type = RENDER_POINTS;
	};

	~Object() {};
};