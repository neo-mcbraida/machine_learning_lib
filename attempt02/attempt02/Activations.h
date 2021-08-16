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
		virtual void SetNodeActivation(vector<Cell*>) = 0;
		void Normalise(vector<Cell*>, float);
	private:
	};

	class Sigmoid : public ActivationFunc{
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Cell*>);
		virtual float Operate(float val);
	private:
	};

	class Relu : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Cell*>);
		virtual float Operate(float val);
	private:
	};

	class Softmax : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Cell*>);
		float AdjustNodeActivation(Cell* node);
		virtual float Operate(float val);
	private:
	};

	class Constant : public ActivationFunc {
	public:
		virtual float DerivActivation(float);
		virtual void SetNodeActivation(vector<Cell*> nodes);
	private:
	};

	class Tanh : public ActivationFunc {
	public:
		virtual float DerivActivation(float val);
		virtual void SetNodeActivation(vector<Cell*> nodes);
		virtual float Operate(float val);
	};
}