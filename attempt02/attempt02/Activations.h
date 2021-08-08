#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "node.h"

using std::vector;
using std::string;

namespace ntwrk {
	class ActivationFunc {
	public:
		virtual float DerivActivation(float) = 0;
		virtual void SetNodeActivation(vector<Node*>) = 0;
		void Normalise(vector<Node*>, float);
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