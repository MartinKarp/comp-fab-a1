#ifdef WIN32
#define NOMINMAX
#endif
#include <app.h>
#include <colormap.h>

#include <ObjectiveFunction.h>
#include <RandomMinimizer.h>
#include <GradientDescentMinimizer.h>
#include <NewtonFunctionMinimizer.h>

#include <iostream>
#include <math.h>
#include <deque>
#include <chrono>
#include <algorithm>
#include <memory>

#include <Eigen/Core>
using Eigen::Vector2f;
using Eigen::Vector2d;
using Eigen::VectorXd;

class RosenbrockFunction : public ObjectiveFunction
{
public:
	virtual double evaluate(const VectorXd& x) const {
		const double &x1 = x[0];
		const double &x2 = x[1];
		return std::pow(a-x1,2.0) + b*std::pow(x2-x1*x1, 2.0);
	}

	virtual void addGradientTo(const VectorXd& x, VectorXd& grad) const {
		const double &x1 = x[0];
		const double &x2 = x[1];
		grad[0] += -2*(a-x1) - 4*b*(x2-x1*x1)*x1;
		grad[1] += 2*b*(x2-x1*x1);
	}

	virtual void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
		const double &x1 = x[0];
		const double &x2 = x[1];
		hessianEntries.push_back(Triplet<double>(0, 0, 2 - 4*b*(x2-x1*x1) + 8*b*x1*x1));
		hessianEntries.push_back(Triplet<double>(1, 0, -4*b*x1));
		hessianEntries.push_back(Triplet<double>(1, 1, 2*b));
	}

	double a = 1, b = 10;
};

class QuadraticFunction : public ObjectiveFunction
{
public:
	virtual double evaluate(const VectorXd& x) const {
		const Vector2d &x2 = x.segment<2>(0);
		return x2.dot(a.cwiseProduct(x2)) + b.dot(x2) + c;
	}

	virtual void addGradientTo(const VectorXd& x, VectorXd& grad) const {
		const Vector2d &x2 = x.segment<2>(0);
		grad.segment<2>(0) += 2.*a.cwiseProduct(x2) + b;
	}

	virtual void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
		for (int i = 0; i < x.size(); ++i)
			hessianEntries.push_back(Triplet<double>(i, i, 2.*a[i]));;
	}

	Vector2d a = {5, 10}, b = {0, 4};
	double c = 0.3;
};

class SineFunction : public ObjectiveFunction
{
public:
	SineFunction() {}

	virtual double evaluate(const VectorXd& x) const {
		double f = 1;
		for (int i = 0; i < x.size(); ++i) {
			f *= sin(x[i]);
		}
		return f;
	}
};

struct MinimizerState
{
	MinimizerState(std::shared_ptr<Minimizer> minimizer, const std::string &name, const NVGcolor &color) : minimizer(minimizer), name(name), color(color) {}
	MinimizerState(Minimizer *minimizer, const std::string &name, const NVGcolor &color) : minimizer(minimizer), name(name), color(color) {}

	std::shared_ptr<Minimizer> minimizer;
	std::string name;
	NVGcolor color;
	std::vector<VectorXd> path;
	bool show = true;
};

class OptimizationDemoApp : public App
{
public:
	OptimizationDemoApp(int width, int height, const char * title, float pixelRatio = 2.f)
		: base(width) {

		setupWindow(width, height, title, pixelRatio);
		clear_color = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);
		lastFrame = std::chrono::high_resolution_clock::now();

		obj = &quadraticFunction;

		// create solvers
		random.reset(new RandomMinimizer());
		gdFixed.reset(new GradientDescentFixedStep(1));
		gdMomentum.reset(new GradientDescentMomentum(1));
		newton.reset(new NewtonFunctionMinimizer(1));
		minimizers.push_back(MinimizerState(random, "Random Minimizer", nvgRGBA(100, 255, 255, 150)));
		minimizers.push_back(MinimizerState(gdFixed, "Gradient Descent Fixed Step Size", nvgRGBA(255, 100, 100, 150)));
		minimizers.push_back(MinimizerState(new GradientDescentVariableStep(1), "Gradient Descent Variable Step Size", nvgRGBA(255, 255, 100, 150)));
		minimizers.push_back(MinimizerState(gdMomentum, "Gradient Descent w/ Momentum", nvgRGBA(100, 255, 100, 150)));
		minimizers.push_back(MinimizerState(newton, "Newton's method",nvgRGBA(255, 100, 255, 150)));
		resetMinimizers();

		// set up `functionValues`, used for plots
		functionValues = new float*[minimizers.size()];
		for (int i = 0; i < minimizers.size(); ++i) {
			functionValues[i] = new float[plotN];
			for (int j = 0; j < plotN; ++j) {
				functionValues[i][j] = 0.f;
			}
		}

		// create iso contours
		generateImage(width, height, true);
	}

	virtual void process(){
		// move image if left mouse button is pressed
		if(mouseDown[GLFW_MOUSE_BUTTON_RIGHT]){
			int dw = (cursorPos[0] - cursorPosDown[0]);
			int dh = (cursorPos[1] - cursorPosDown[1]);
			translation[0] += dw/(double)base;
			translation[1] -= dh/(double)base;
			cursorPosDown[0] = cursorPos[0];
			cursorPosDown[1] = cursorPos[1];

			translateImage(dw, dh);
		}

		// run at 60fps, or in slow mo
		std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
		if(std::chrono::duration_cast<std::chrono::milliseconds>(now-lastFrame).count() > ((slowMo) ? 320 : 16)){

			// call all minimizers
			if(isMinimize){
				int i = 0;
				for(auto &m : minimizers) {
					if(m.path.size() > 0) {
						VectorXd x = m.path[m.path.size()-1];
						m.minimizer->minimize(obj, x);
						m.path.push_back(x);
						// record function values for plots
						float f = (float)obj->evaluate(x);
						functionValues[i++][plotCounter] = f;
					}
				}
				plotCounter = (plotCounter+1) % plotN;
				if(plotCounter == 0) plotStatic = false;
			}

			lastFrame = now;
		}
	}

	virtual void drawScene() {

		// this is the GUI
		{
			ImGui::Begin("Assignement 1");

			ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "left mouse:  start optimization");
			ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "right mouse: move function landscape");
			ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "mouse wheel: zoom function landscape");
			ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "space bar:   play/pause optimization");

			ImGui::Separator();

			const char * items[3] = {"Quadratic Function", "Rosenbrock", "Sine Function"};
			if(ImGui::Combo("Function", &currentFctn, items, 3)){
				if(currentFctn == 0) obj = &quadraticFunction;
				else if(currentFctn == 1) obj = &rosenbrockFunction;
				else if(currentFctn == 2) obj = &sineFunction;
				generateImage(display_w, display_h, true);
				isMinimize = false;
			}

			ImGui::Checkbox("play [space]", &isMinimize);
			ImGui::Checkbox("Slow motion", &slowMo);

			for (auto &m : minimizers) {
				ImGui::PushID(m.name.c_str());
				ImGui::ColorButton(m.name.c_str(), ImVec4(m.color.r, m.color.g, m.color.b, 1.0));
				ImGui::SameLine();
				ImGui::Text("%s", m.name.c_str());
				ImGui::SameLine();
				ImGui::Checkbox("", &m.show);
				ImGui::PopID();
			}

			const double smin = 0.0, smax = 1.0;
			ImGui::SliderScalar("GD Fixed: step size", ImGuiDataType_Double, &gdFixed->stepSize, &smin, &smax);
			ImGui::SliderScalar("GD Momentum: alpha", ImGuiDataType_Double, &gdMomentum->alpha, &smin, &smax);
			ImGui::InputDouble("Newton: regularizer", &newton->reg);

			// plots
			if(ImGui::CollapsingHeader("Plot"))	{
				const int n = minimizers.size();
				const char ** names = new const char*[n];
				ImColor *colors = new ImColor[n];
				int i = 0;
				int p = plotN - plotZoom;
				for (const auto &m : minimizers) {
					names[i] = m.name.c_str();
					colors[i] = ImColor(m.color.r, m.color.g, m.color.b);
					i++;
				}

				int plotNZoom = plotZoom;
				ImGui::PlotMultiLines("Function values", n, names, colors,
									  [](const void *data, int idx)->float { return ((const float*)data)[idx]; },
				((const void *const *)functionValues), plotN, plotNZoom, ((plotStatic) ? 0 : plotCounter) + plotPan, min, max, ImVec2(0, 150));

				ImGui::SliderInt("plot zoom", &plotZoom, plotN, 0);
				ImGui::SliderInt("plot pan", &plotPan, 0, plotN - plotNZoom);
			}

			if(ImGui::CollapsingHeader("Landscape Look")){
				if(ImGui::InputInt("# of contours", &nContours))
					generateImage(display_w, display_h);

				if(ImGui::Button("Recompute contours"))
					generateImage(display_w, display_w, true);

				if(ImGui::Checkbox("show contour edges", &showContourEdges))
					generateImage(display_w, display_h);
			}

			ImGui::End();
		}

		// draw contour lines
		nvgBeginPath(vg);
		NVGpaint imgPaint = nvgImagePattern(vg, 0.f, 0.f, display_w, display_h, 0.f, img, 1.0f);
		nvgRect(vg, 0.f, 0.f, display_w, display_h);
		nvgFillPaint(vg, imgPaint);
		nvgFill(vg);

		// draw path for each minimizer
		for(const auto &m : minimizers)
		{
			if(m.path.size() > 0 && m.show){
				nvgBeginPath(vg);
				nvgMoveTo(vg, toScreen(m.path[0][0], 0), toScreen(m.path[0][1], 1));
				for(const auto &p : m.path)
					nvgLineTo(vg, toScreen(p[0], 0), toScreen(p[1], 1));
				nvgStrokeColor(vg, m.color);
				nvgStrokeWidth(vg, 4.f);
				nvgStroke(vg);

				nvgBeginPath(vg);
				for(const auto &p : m.path)
					nvgCircle(vg, toScreen(p[0], 0), toScreen(p[1], 1), 5);
				nvgStrokeColor(vg, m.color);
				nvgStrokeWidth(vg, 2.f);
				nvgStroke(vg);
				nvgFillColor(vg, m.color);
				nvgFill(vg);
			}
		}

	}

protected:	
	virtual void keyPressed(int key, int mods) {
		// play / pause with space bar
		if(key == GLFW_KEY_SPACE)
			isMinimize = !isMinimize;
	}

	virtual void mousePressed(int button) {
		// restart optimization when left mouse clicked
		if(button == GLFW_MOUSE_BUTTON_LEFT){
			VectorXd x = fromScreen(cursorPos[0], cursorPos[1]);
			float f = (float)obj->evaluate(x);
			plotStatic = true;

			resetMinimizers(x);

			for(auto & m : minimizers)
				m.path.push_back(x);

			for (int i = 0; i < minimizers.size(); ++i) {
				functionValues[i][0] = f;
				for (int j = 1; j < plotN; ++j) {
					functionValues[i][j] = min;
				}
			}
			plotCounter = 1;

			isMinimize = true;
		}
		else {
			cursorPosDown[0] = cursorPos[0];
			cursorPosDown[1] = cursorPos[1];
		}
	}

	virtual void mouseReleased(int button) {
	}

	virtual void scrollWheel(double xoffset, double yoffset) {
		double zoomOld = zoom;
		zoom *= std::pow(1.10, yoffset);
		for (int dim = 0; dim < 2; ++dim) {
			double c = cursorPos[dim]/(double) ((dim == 0) ? base : -base);
			translation[dim] = c - zoomOld/zoom * (c-translation[dim]);
		}
		generateImage(display_w, display_h);
	}

	virtual void windowResized(int w, int h) {
		generateImage(w, h);
	}


private:
	void generateImage(int w, int h, bool recompute = false) {

		// find min/max
		if(recompute){
			min = HUGE_VAL; max = -HUGE_VAL;
			for (int j = 0; j < h; ++j) {
				for (int i = 0; i < w; ++i) {
					double f = obj->evaluate(fromScreen(i, j, base, base));
					min = std::min(min, f);
					max = std::max(max, f);
				}
			}
		}

		if(imgData != nullptr)
			free(imgData);
		imgData = new unsigned char[w*h*4];
		unsigned char* px = imgData;
		for (int j = 0; j < h; ++j) {
			for (int i = 0; i < w; ++i) {
				// sample function at i,j
				double f = obj->evaluate(fromScreen(i, j, base, base));
				// use log-scale, +1 to prevent log(0)
				f = log(f - min + 1) / log(max - min + 1);

				// put into 'buckets'
				f = floor(f*(double)nContours) / (double)nContours;

				// map to a color
				float r, g, b;
				colorMapColor(f, r, g, b);
				px[0] = (unsigned char)(255.f*r);
				px[1] = (unsigned char)(255.f*g);
				px[2] = (unsigned char)(255.f*b);
				px[3] = 150;
				px += 4; // advance to next pixel
			}
		}

		if(showContourEdges){
			px = imgData;
			for (int j = 0; j < h; ++j) {
				for (int i = 0; i < w; ++i) {
					for (int k = 0; k < 3; ++k) {
						if((i < w-1 && px[k] != px[4+k]) || (j<h-1 && px[k] != px[4*w+k])){
							px[0] = 0;
							px[1] = 0;
							px[2] = 0;
							px[3] = 150;
							break;
						}
					}
					px += 4;
				}
			}
		}

		if(img != -1)
			nvgDeleteImage(vg, img);
		img = nvgCreateImageRGBA(vg, w, h, 0, imgData);
	}

	void translateImage(int dw, int dh, bool recompute = false) {

		const int w = display_w;
		const int h = display_h;

		// find min/max
		if(recompute){
			min = HUGE_VAL; max = -HUGE_VAL;
			for (int j = 0; j < h; ++j) {
				for (int i = 0; i < w; ++i) {
					double f = obj->evaluate(fromScreen(i, j, base, base));
					min = std::min(min, f);
					max = std::max(max, f);
				}
			}
		}

		int wm = std::max(0, dw);
		int wp = std::min(w, w+dw);
		int hm = std::max(0, dh);
		int hp = std::min(h, h+dh);

		unsigned char * imgDataNew = new unsigned char[w*h*4];
		unsigned char* px = imgDataNew;
		for (int j = 0; j < h; ++j) {
			for (int i = 0; i < w; ++i) {

				unsigned char r, g, b, a;
				if(i >= wm && i < wp && j >= hm && j < hp){
					int idx = (w*(j-dh) + i-dw)*4;
					r = imgData[idx + 0];
					g = imgData[idx + 1];
					b = imgData[idx + 2];
					a = imgData[idx + 3];
				}
				else{
					// sample function at i,j
					double f = obj->evaluate(fromScreen(i, j, base, base));
					// use log-scale, +1 to prevent log(0)
					f = log(f - min + 1) / log(max - min + 1);

					// put into 'buckets'
					f = floor(f*(double)nContours) / (double)nContours;
					float rf, gf, bf;
					colorMapColor(f, rf, gf, bf);
					r = (unsigned char)255.f*rf;
					g = (unsigned char)255.f*gf;
					b = (unsigned char)255.f*bf;
					a = 150;
				}

				px[0] = r;
				px[1] = g;
				px[2] = b;
				px[3] = a;
				px += 4; // advance to next pixel
			}
		}

		if(imgData != nullptr)
			free(imgData);
		imgData = imgDataNew;

		if(showContourEdges){
			px = imgData;
			for (int j = 0; j < h; ++j) {
				for (int i = 0; i < w; ++i) {
					for (int k = 0; k < 3; ++k) {
						if((i < w-1 && px[k] != px[4+k]) || (j<h-1 && px[k] != px[4*w+k])){
							px[0] = 0;
							px[1] = 0;
							px[2] = 0;
							px[3] = 150;
							break;
						}
					}
					px += 4;
				}
			}
		}

		if(img != -1)
			nvgDeleteImage(vg, img);
		img = nvgCreateImageRGBA(vg, w, h, 0, imgData);
	}

	void resetMinimizers(const VectorXd &x = VectorXd()){

		// reset random minimizer
		random->searchDomainMax = fromScreen(0,0);
		random->searchDomainMin = fromScreen(window_w,window_h);
		random->fBest = (x.size() == 0) ? HUGE_VAL : obj->evaluate(x);
		random->xBest = x;

		// reset GD Momentum
		gdMomentum->gradient.setZero();

		// remove all recorded paths
		for (auto &m : minimizers) {
			m.path.clear();
		}
	}

	VectorXd fromScreen(int i, int j, int w, int h) const {
		VectorXd x(2);
		x[0] = ((double)i/(double)w - translation[0])*zoom;
		x[1] = (-(double)j/(double)h - translation[1])*zoom;
		return x;
	}

	template<class S>
	VectorXd fromScreen(S i, S j) const {
		return fromScreen((double)i, (double)j, base, base);
	}

	double toScreen(double s, int dim) const {
		return (s/zoom + translation[dim]) * (double)((dim == 0) ? base : -base);
	}

private:
	// objective functions
	QuadraticFunction quadraticFunction;
	RosenbrockFunction rosenbrockFunction;
	SineFunction sineFunction;
	ObjectiveFunction *obj = nullptr;
	int currentFctn = 0;

	// minimizers
	std::vector<MinimizerState> minimizers;
	std::shared_ptr<RandomMinimizer> random;
	std::shared_ptr<GradientDescentFixedStep> gdFixed;
	std::shared_ptr<GradientDescentMomentum> gdMomentum;
	std::shared_ptr<NewtonFunctionMinimizer> newton;
	bool isMinimize = false;
	bool slowMo = false;

	// user interface
	double cursorPosDown[2];
	double min, max;
	double translation[2] = {0.5, -0.5};
	double zoom = 10;
	int base;
	int nContours = 10;
	unsigned char * imgData = nullptr;
	int img;
	bool showContourEdges = false;
	std::chrono::high_resolution_clock::time_point lastFrame;

	// plot
	static const int plotN = 200;
	int plotZoom = plotN;
	int plotPan = 0;
	int plotCounter = 0;
	bool plotStatic = true;
	float **functionValues;
};

int main(int, char**)
{
	// If you have high DPI screen settings, you can change the pixel ratio
	// accordingly. E.g. for 200% scaling use `pixelRatio = 2.f`
	float pixelRatio = 1.f;
	OptimizationDemoApp app(1080, 720, "Assignement 1", pixelRatio);
	app.run();

	return 0;
}
