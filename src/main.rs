use encase::{ShaderSize, ShaderType};
use geometric_algebra::{pga3d::*, *};
use glam::Vec3;
use winit::{
	dpi::{PhysicalPosition, PhysicalSize},
	event::{
		DeviceEvent, ElementState, Event, MouseScrollDelta::LineDelta, RawKeyEvent, WindowEvent,
	},
	event_loop::EventLoop,
	keyboard::{KeyCode, PhysicalKey},
	window::{Window, WindowBuilder},
};

// switch between 3D and 4D (TODO implement 4D)
const N: usize = 3;
type VecN = glam::Vec3;
type BVecN = glam::BVec3;
const WGSL_GEOMETRIC_ALGEBRA: &str = include_str!("../geometric_algebra/src/pga3d.wgsl");
const BASIS: [Hyperplane; N + 1] = [
        Hyperplane::new(0.0, 0.0, 0.0, 1.0), // e0 (i put the dual element at the end)
	Hyperplane::new(1.0, 0.0, 0.0, 0.0), // e1
	Hyperplane::new(0.0, 1.0, 0.0, 0.0), // e2
	Hyperplane::new(0.0, 0.0, 1.0, 0.0), // e3
];
const ORIGIN: Point = Point::new(0.0, 0.0, 0.0, 1.0); // BASIS[0].dual() is non const..
// comment these out when N > 3
type MeetLine = MeetJoinLine; // rates (velocity), accelerations
type JoinLine = MeetJoinLine; // forques, momenta



// don't change dimensions after this
const MAX_POINT_LIGHTS: usize = 1;
const MAX_BODIES: usize = 16;

static SHADER_WGSL: &str = const_format::concatcp!(
	WGSL_GEOMETRIC_ALGEBRA,
	"const N = ",
	N,
	";alias vecN = vec",
	N,
	"f;const MAX_POINT_LIGHTS = ",
	MAX_POINT_LIGHTS,
	";const MAX_BODIES = ",
	MAX_BODIES,
	";const ORIGIN = Point(vecN(), 1.0);",
	include_str!("shader.wgsl"),
);

#[derive(Debug, Clone)]
struct Camera {
	motion: Motor,
	// fov: f32,
	up: Point,
	forward: Point,
	left: Point,
}

#[derive(Debug, ShaderType)]
struct ShaderPoint {
	#[align(16)]
	g0: VecN,
	// #[align(16)]
	g1: f32,
}
impl From<Point> for ShaderPoint {
	fn from(point: Point) -> Self {
		Self {
			g0: {
				let mut g0 = VecN::ZERO;
				for i in 0..N {
					g0[i] = point[i];
				}
				g0
			},
			g1: point[N],
		}
	}
}
#[derive(Debug, ShaderType)]
struct ShaderMotor {
	#[align(16)]
	g0: VecN,
	#[align(16)]
	g1: VecN,
	#[align(16)]
	g2: glam::Vec2,
}
impl From<Motor> for ShaderMotor {
	fn from(motor: Motor) -> Self {
		Self {
			g0: {
				let mut g0 = VecN::ZERO;
				for i in 0..N {
					g0[i] = motor[i];
				}
				g0
			},
			g1: {
				let mut g1 = VecN::ZERO;
				for i in 0..N {
					g1[i] = motor[i + N];
				}
				g1
			},
			g2: glam::Vec2::new(motor[N + N], motor[N + N + 1]),
		}
	}
}
#[derive(Debug, ShaderType)]
struct ShaderCamera {
	#[align(16)]
	motion: ShaderMotor,
	#[align(16)]
	up: ShaderPoint,
	#[align(16)]
	forward: ShaderPoint,
	#[align(16)]
	left: ShaderPoint,
}
impl From<Camera> for ShaderCamera {
	fn from(camera: Camera) -> Self {
		ShaderCamera {
			motion: camera.motion.into(),
			up: camera.up.into(),
			forward: camera.forward.into(),
			left: camera.left.into(),
		}
	}
}
#[derive(Debug, Clone)]
struct PointLight {
	color: Vec3,
	position: Point,
}
#[derive(Debug, ShaderType)]
struct ShaderPointLight {
	#[align(16)]
	color: Vec3,
	#[align(16)]
	position: ShaderPoint,
}
impl From<PointLight> for ShaderPointLight {
	fn from(point_light: PointLight) -> Self {
		Self {
			color: point_light.color,
			position: point_light.position.into(),
		}
	}
}
// no need for a separate ShaderMaterial
#[derive(Clone, Debug, ShaderType)]
struct Material {
	ambient: Vec3,
	diffuse: Vec3,
}
#[derive(Clone, Debug)]
enum Shape {
        None,
	Sphere(f32),
	Box(VecN),
}
#[derive(Debug, ShaderType)]
struct ShaderShape {
	#[align(16)]
	params: VecN,
}
impl From<Shape> for ShaderShape {
	fn from(shape: Shape) -> Self {
		Self {
			params: match shape {
                                Shape::None => VecN::ZERO,
				Shape::Sphere(radius) => VecN::X * radius,
				Shape::Box(half_widths) => half_widths,
			},
		}
	}
}
#[derive(Clone, Debug)]
struct Body {
	motion: Motor,   // "position": M * template "position" * M.reverse == world "position"
	rate: MeetLine,      // template "velocity"
	prev_rate: MeetLine, // for velocity-based contact resolution?
	shape: Shape,
	material: Material,
}
#[derive(Debug, ShaderType)]
struct ShaderBody {
	#[align(16)]
	motion: ShaderMotor,
	#[align(16)]
	shape: ShaderShape,
	#[align(16)]
	material: Material,
}
impl From<Body> for ShaderBody {
	fn from(body: Body) -> Self {
		Self {
			motion: body.motion.into(),
			shape: body.shape.into(),
			material: body.material,
		}
	}
}

#[derive(Debug)]
struct AppState {
	// controls
	key_plus: BVecN,
	key_minus: BVecN,
	mouse_delta: VecN,
	window_size: PhysicalSize<u32>,
	// physics
	time: std::time::Instant,
	speed: f32,
	gravity_accel: MeetLine,
	place_distance: f32,
	place_shape: Shape,
	// render
	camera: Camera,
	global_light_color: Vec3,
	point_lights: [PointLight; MAX_POINT_LIGHTS],
	num_bodies: usize,
	bodies: [Body; MAX_BODIES],
	floor: Hyperplane,
}
#[derive(Debug, ShaderType)]
struct ShaderState {
	#[align(16)]
	camera: ShaderCamera,
	#[align(16)]
	global_light_color: Vec3,
	#[align(16)]
	point_lights: [ShaderPointLight; MAX_POINT_LIGHTS],
	#[align(16)]
	bodies: [ShaderBody; MAX_BODIES],
}
impl From<&AppState> for ShaderState {
	fn from(state: &AppState) -> Self {
		Self {
			camera: ShaderCamera::from(state.camera.clone()),
			global_light_color: state.global_light_color,
			point_lights: state.point_lights.clone().map(ShaderPointLight::from),
			bodies: state.bodies.clone().map(ShaderBody::from),
		}
	}
}

impl AppState {
	fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
		let mut buffer = encase::UniformBuffer::new(Vec::new());
		buffer
			.write(&ShaderState::from(self))
			.expect("encase::UniformBuffer::write error");
		Ok(buffer.into_inner())
	}
	fn simulate(&mut self, dt: std::time::Duration) {
		/*
		q[i+1] = q[i] + h qdot[i]
		qdot[i+1] = qdot[i] - h * Minv * dV(q[i+1])
		===
		q += h * qdot
		qdot += h * Minv * F(q)
		 */
		/*
		https://enki.ws/ganja.js/examples/pga_dyn.html
		d/dt motion = -0.5 * motion * rate
		d/dt rate = -0.5 * (rate.dual * rate - rate * rate.dual).undual (what is undual?)
		 */
		let h = dt.as_secs_f32();
		// physics simulation
		{
			for (i, body) in self.bodies[..self.num_bodies].iter_mut().enumerate() {
                                dbg!(&body.motion);
                                if true{break;}
				let forque = body.motion.reversal().transformation(self.gravity_accel).dual();
				body.motion += body.motion.geometric_product(body.rate) * (-0.5 * h);
				body.rate += (((body.rate.dual().geometric_product(body.rate))
					- body.rate.geometric_product(body.rate.dual()))
				.dual()
				.reversal() * -0.5)
					.into();
				/*
				let mut mnext = body.motion - body.motion * body.rate * (0.5 * h);
				match body.shape {
					// collisions
					Shape::Sphere(r) => {
						// ?? do some projection shit
						let y = body.motion.transformation(ORIGIN).regressive_product(self.floor);
					}
					Shape::Box(half_widths) => {
						// more projection shit
					}
				}
				body.prev_rate = body.rate;
				body.motion = mnext;
				// https://enki.ws/ganja.js/examples/pga_dyn.html#Collision_Response
				body.rate += (body.rate.dual()*body.rate - body.rate*body.rate.dual()).dual().inverse() * -0.5;
				// body.rate += dbg!(body.motion.reversal().transformation(self.gravity).dual() * h);
				// wtf
				 */
			}
			/*
			for i in 0..self.num_spheres {
				let mut qnext = self.shader_state.spheres[i].center + h * self.sphere_vels[i];
				if qnext.x < 0.0 {
					self.sphere_vels[i].x = -self.sphere_prev_vels[i].x;
					qnext = self.shader_state.spheres[i].center + h * self.sphere_vels[i];
				}
				self.sphere_prev_vels[i] = self.sphere_vels[i];
				self.shader_state.spheres[i].center = qnext;
				self.sphere_vels[i].x += hg;
			}
			for i in 0..self.num_boxes {
				let mut qnext = self.shader_state.boxes[i].center + h * self.box_vels[i];
				if qnext.x-self.shader_state.boxes[i].half_widths.x < 0.0 {
					self.box_vels[i].x = -self.box_prev_vels[i].x;
					qnext = self.shader_state.boxes[i].center + h * self.box_vels[i];
				}
				self.box_prev_vels[i] = self.box_vels[i];
				self.shader_state.boxes[i].center = qnext;
				self.box_vels[i].x += hg;
			}
			 */
		}
		// camera update
		let t = {
			let t = -0.5 * h * self.speed * (VecN::from(self.key_plus) - VecN::from(self.key_minus));
			Translator::new(1.0, t.x, t.y, t.z)
		};
		let r = Rotor::new(
			1.0,
			self.mouse_delta.x * 1e-3,
			self.mouse_delta.z * 1e-1,
			-self.mouse_delta.y * 1e-3,
		)
		.signum();
		self.mouse_delta = VecN::ZERO;
		self.camera.motion = self
			.camera
			.motion
			.geometric_product(r)
			.geometric_product(t)
			.signum();
	}
	fn try_place_body(&mut self) {
		if self.num_bodies == MAX_BODIES {
			return;
		}
		let rand_color = Vec3::new(rand::random(), rand::random(), rand::random());
		let motion = self.camera.motion.geometric_product(Translator::new(
                        1.0,
			0.0,
			-0.5 * self.place_distance,
			0.0,
		));
		let rand_around_1 = || (rand::random::<f32>() - 0.5) + 1.0;
		self.bodies[self.num_bodies] = Body {
			motion,
			rate: motion
				.reversal()
                                // TODO why dual? re-read eq. 2.25 from https://bivector.net/PGAdyn.pdf
				.transformation(
                                    dbg!(BASIS[0].outer_product(BASIS[1]))
                                ),
			prev_rate: MeetLine::zero(),
			shape: if rand::random::<f32>() < 0.5 {
				Shape::Sphere(rand_around_1())
			} else {
				Shape::Box(VecN::from_array(std::array::from_fn(|_| rand_around_1())))
			},
			material: Material {
				ambient: rand_color,
				diffuse: rand_color,
			},
		};
                self.bodies[self.num_bodies].rate = MeetLine::zero(); // TODO DELETE THIS
                dbg!(&self.bodies[self.num_bodies]);
		self.num_bodies += 1;
	}
	#[cfg(any())]
	fn shader_state(&mut self) -> &ShaderState {
		let cam_pos = self.camera.transformation(Point::new(1.0, 0.0, 0.0, 0.0));
		self.shader_state.camera.position = VecN::new(cam_pos[1], cam_pos[2], cam_pos[3]);
		let cam_up =
			self.cam
				.transformation(Point::new(0.0, self.window_size.height as f32, 0.0, 0.0));
		self.shader_state.camera.up = VecN::new(cam_up[1], cam_up[2], cam_up[3]);
		let cam_forward = self.camera.transformation(Point::new(
			0.0,
			0.0,
			((self.window_size.width * self.window_size.width
				+ self.window_size.height * self.window_size.height) as f32)
				.sqrt() / (0.5 * self.fov).tan(),
			0.0,
		));
		self.shader_state.camera.forward =
			VecN::new(cam_forward[1], cam_forward[2], cam_forward[3]);
		let cam_left =
			self.cam
				.transformation(Point::new(0.0, 0.0, 0.0, self.window_size.width as f32));
		self.shader_state.camera.left = VecN::new(cam_left[1], cam_left[2], cam_left[3]);
		&self.shader_state
	}
}

async fn run(event_loop: EventLoop<()>, window: Window) {
	let mut app_state = AppState {
		// controls
		key_plus: BVecN::FALSE,
		key_minus: BVecN::FALSE,
		mouse_delta: VecN::ZERO,
		window_size: window.inner_size(),
		// physics
		time: std::time::Instant::now(),
		speed: 1.0,
		gravity_accel: dbg!(BASIS[0].outer_product(BASIS[1])) * 10.0,
		place_distance: 4.0,
		place_shape: Shape::Sphere(1.0),
		// render
		camera: dbg!(Camera {
			motion: Motor::one(),
			up: BASIS[1].dual(),
			forward: BASIS[2].dual(), // TODO should this be a plane?
			left: BASIS[3].dual(),
                        // we don't need an additional camera param in 4D
                        // because the camera window is always 2d
		}),
		global_light_color: Vec3::ONE * 0.125,
		point_lights: [PointLight {
			color: Vec3::ONE,
			position: ORIGIN + BASIS[1].dual() * 128.0,
		}; MAX_POINT_LIGHTS],
		floor: BASIS[1],
		num_bodies: 0,
		bodies: std::array::from_fn(|_| Body {
			motion: Motor::zero(),
			rate: MeetLine::zero(),
			prev_rate: MeetLine::zero(),
			shape: Shape::None,
			material: Material {
				ambient: Vec3::ONE,
				diffuse: Vec3::ONE,
			},
		}),
	};
	app_state.window_size.width = app_state.window_size.width.max(1);
	app_state.window_size.height = app_state.window_size.height.max(1);

	let instance = wgpu::Instance::default();

	let surface = instance.create_surface(&window).unwrap();
	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::default(),
			force_fallback_adapter: false,
			// Request an adapter which can render to our surface
			compatible_surface: Some(&surface),
		})
		.await
		.expect("Failed to find an appropriate adapter");

	// Create the logical device and command queue
	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				required_features: wgpu::Features::empty(),
				// Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
				required_limits: wgpu::Limits::downlevel_webgl2_defaults()
					.using_resolution(adapter.limits()),
			},
			None,
		)
		.await
		.expect("Failed to create device");

	// Load the shaders from disk
	let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
		label: None,
		source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_WGSL)),
	});

	let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
		label: None,
		size: dbg!(ShaderState::SHADER_SIZE.get()),
		usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});
	let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: None,
		entries: &[wgpu::BindGroupLayoutEntry {
			binding: 0,
			visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Uniform,
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		}],
	});
	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: None,
		layout: &bind_group_layout,
		entries: &[wgpu::BindGroupEntry {
			binding: 0,
			resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
				buffer: &uniform_buffer,
				offset: 0,
				size: None,
			}),
		}],
	});

	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: None,
		bind_group_layouts: &[&bind_group_layout],
		push_constant_ranges: &[],
	});

	let swapchain_capabilities = surface.get_capabilities(&adapter);
	let swapchain_format = swapchain_capabilities.formats[0];

	let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: None,
		layout: Some(&pipeline_layout),
		vertex: wgpu::VertexState {
			module: &shader,
			entry_point: "vs_main",
			buffers: &[],
		},
		primitive: wgpu::PrimitiveState::default(),
		depth_stencil: None,
		multisample: wgpu::MultisampleState::default(),
		multiview: None,
		fragment: Some(wgpu::FragmentState {
			module: &shader,
			entry_point: "fs_main",
			targets: &[Some(swapchain_format.into())],
		}),
	});

	let mut config = surface
		.get_default_config(
			&adapter,
			app_state.window_size.width,
			app_state.window_size.height,
		)
		.unwrap();
	config.present_mode = wgpu::PresentMode::AutoVsync;
	surface.configure(&device, &config);

	let window = &window;
	window.set_cursor_visible(false);
	event_loop
		.run(move |event, target| {
			// Have the closure take ownership of the resources.
			// `event_loop.run` never returns, therefore we must do this to ensure
			// the resources are properly cleaned up.
			let _ = (&instance, &adapter, &shader, &pipeline_layout);

			match event {
				Event::DeviceEvent { event, .. } => match event {
					DeviceEvent::MouseMotion { delta: (x, y) } => {
						app_state.mouse_delta.x += x as f32;
						app_state.mouse_delta.y += y as f32;
						window
							.set_cursor_position(PhysicalPosition::new(
								app_state.window_size.width >> 1,
								app_state.window_size.height >> 1,
							))
							.expect("unable to set cursor position");
					}
					DeviceEvent::MouseWheel { delta } => {
						if let LineDelta(_, y) = delta {
							let scale = 0.99f32.powi(y as i32);
							app_state.camera.up *= scale;
							app_state.camera.left *= scale;
						// fov ??
						} else {
							dbg!(delta);
						}
					}
					DeviceEvent::Button { button, state } => {
						if state == ElementState::Pressed {
							match button {
								0 => {
									// mouse left
									app_state.try_place_body();
								}
								1 => {
									// mouse right
								}
								2 => {
									// mouse middle
								}
								3 => {
									// mouse back
									app_state.mouse_delta.z += 1.0;
								}
								4 => {
									// mouse front
									app_state.mouse_delta.z -= 1.0;
								}
								_ => {
									dbg!(button);
								}
							}
						}
					}
					DeviceEvent::Key(RawKeyEvent {
						physical_key,
						state,
					}) => {
						if let PhysicalKey::Code(key_code) = physical_key {
							let pressed = state == ElementState::Pressed;
							match key_code {
								KeyCode::Space => app_state.key_plus.x = pressed,
								KeyCode::ShiftLeft => app_state.key_minus.x = pressed,
								KeyCode::KeyW => app_state.key_plus.y = pressed,
								KeyCode::KeyS => app_state.key_minus.y = pressed,
								KeyCode::KeyA => app_state.key_plus.z = pressed,
								KeyCode::KeyD => app_state.key_minus.z = pressed,
								_ => {}
							}
						}
					}
					_ => {}
				},
				Event::WindowEvent { event, .. } => match event {
					WindowEvent::Resized(new_size) => {
						// Reconfigure the surface with the new size
						config.width = new_size.width.max(1);
						config.height = new_size.height.max(1);
						app_state.window_size.width = config.width;
						app_state.window_size.height = config.height;
						let diag = ((config.width * config.width + config.height * config.height)
							as f32)
							.sqrt()
							.recip();
						app_state.camera.up = BASIS[1].dual() * (config.height as f32 * diag);
						app_state.camera.left = BASIS[3].dual() * (config.width as f32 * diag);
						surface.configure(&device, &config);
						// On macOS the window needs to be redrawn manually after resizing
						window.request_redraw();
					}
					WindowEvent::RedrawRequested => {
						let time = std::time::Instant::now();
						let dt = time - app_state.time;
						// dbg!(dt.as_secs_f32().recip());
						app_state.time = time;
						app_state.simulate(dt);
						let frame = surface
							.get_current_texture()
							.expect("Failed to acquire next swap chain texture");
						let view = frame
							.texture
							.create_view(&wgpu::TextureViewDescriptor::default());
						queue.write_buffer(
							&uniform_buffer,
							0,
							&app_state.as_wgsl_bytes().expect(
								"Error in encase translating AppState struct to WGSL bytes.",
							),
						);
						let mut encoder =
							device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
								label: None,
							});
						{
							let mut rpass =
								encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
									label: None,
									color_attachments: &[Some(wgpu::RenderPassColorAttachment {
										view: &view,
										resolve_target: None,
										ops: wgpu::Operations {
											load: wgpu::LoadOp::Load,
											store: wgpu::StoreOp::Store,
										},
									})],
									depth_stencil_attachment: None,
									timestamp_writes: None,
									occlusion_query_set: None,
								});
							rpass.set_pipeline(&render_pipeline);
							rpass.set_bind_group(0, &bind_group, &[]);
							rpass.draw(0..3, 0..2);
						}

						queue.submit(Some(encoder.finish()));
						frame.present();
						window.request_redraw();
					}
					WindowEvent::CloseRequested => target.exit(),
					_ => {}
				},
				_ => {}
			}
		})
		.unwrap();
}

pub fn main() {
	let event_loop = EventLoop::new().unwrap();
	#[allow(unused_mut)]
	let mut builder = WindowBuilder::new();
	#[cfg(target_arch = "wasm32")]
	{
		use wasm_bindgen::JsCast;
		use winit::platform::web::WindowBuilderExtWebSys;
		let canvas = web_sys::window()
			.unwrap()
			.document()
			.unwrap()
			.get_element_by_id("canvas")
			.unwrap()
			.dyn_into::<web_sys::HtmlCanvasElement>()
			.unwrap();
		builder = builder.with_canvas(Some(canvas));
	}
	let window = builder.build(&event_loop).unwrap();

	#[cfg(not(target_arch = "wasm32"))]
	{
		env_logger::init();
		pollster::block_on(run(event_loop, window));
	}
	#[cfg(target_arch = "wasm32")]
	{
		std::panic::set_hook(Box::new(console_error_panic_hook::hook));
		console_log::init().expect("could not initialize logger");
		wasm_bindgen_futures::spawn_local(run(event_loop, window));
	}
}
