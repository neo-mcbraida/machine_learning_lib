#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "node.h"

using std::vector;
using std::string;
//using ntwrk::Node;

namespace ntwrk {
	class ActivationFunc {
	public:
		virtual float DerivActivation(float) = 0;
		virtual void SetNodeActivation(vector<Node*>) = 0;
		//virtual float AdjustNodeActivation(Node*);
	private:
	};

	class Sigmoid : public ActivationFunc{
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Node*>);
	private:
	};

	class Relu : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Node*>);
	private:
	};

	class Softmax : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Node*>);
		float AdjustNodeActivation(Node* node);
	private:
	};

	class Constant : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Node*> nodes);
	private:
	};
}