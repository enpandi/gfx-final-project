use cfg_if::cfg_if;
use encase::{ShaderSize, ShaderType};
use geometric_algebra::*;
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

cfg_if! {
	if #[cfg(feature = "four_d")] {
		use geometric_algebra::pga4d::*;
	} else {
		use geometric_algebra::pga3d::*;
	}
}
// switch between 3D and 4D (TODO implement 4D)
cfg_if! {
	if #[cfg(feature = "four_d")] {
		const N: usize = 4;
		type VecN = glam::Vec4;
		type BVecN = glam::BVec4;
		const WGSL_GEOMETRIC_ALGEBRA: &str = include_str!("../geometric_algebra/src/pga4d.wgsl");
		const BASIS: [Plane; N + 1] = [
			Plane::new(0.0, 0.0, 0.0, 0.0,1.0), // e0 (i put the dual element at the end)
			Plane::new(1.0, 0.0, 0.0, 0.0,0.0), // e1
			Plane::new(0.0, 1.0, 0.0, 0.0,0.0), // e2
			Plane::new(0.0, 0.0, 1.0, 0.0,0.0), // e3
			Plane::new(0.0, 0.0, 0.0, 1.0,0.0), // e3
		];
		const ORIGIN: Point = Point::new(0.0, 0.0, 0.0, 0.0, 1.0);
	} else {

		const N: usize = 3;
		type VecN = glam::Vec3;
		type BVecN = glam::BVec3;
		const WGSL_GEOMETRIC_ALGEBRA: &str = include_str!("../geometric_algebra/src/pga3d.wgsl");
		const BASIS: [Plane; N + 1] = [
			Plane::new(0.0, 0.0, 0.0, 1.0), // e0 (i put the dual element at the end)
			Plane::new(1.0, 0.0, 0.0, 0.0), // e1
			Plane::new(0.0, 1.0, 0.0, 0.0), // e2
			Plane::new(0.0, 0.0, 1.0, 0.0), // e3
		];
		const ORIGIN: Point = Point::new(0.0, 0.0, 0.0, 1.0);
		// BASIS[0].dual() is non const
		// also this library uses different dual conventions from most of the literature i'm referencing..

		type MeetLine = MeetJoinLine; // rates (velocity), accelerations
		type JoinLine = MeetJoinLine; // forques, momenta
	}
}

// don't change dimensions after this
// const PSS: Pseudoscalar = Pseudoscalar::new(1.0); // generator is stupid can't handle PSS
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

const SHADER_POINT_NUM_GROUPS: usize = (N + 4) / 4;
// const SHADER_POINT_NUM_GROUPS: usize =
// 	(std::mem::size_of::<Point>() / std::mem::size_of::<f32>() + 3) / 4;
#[derive(Debug, ShaderType)]
struct ShaderPoint {
	#[align(16)]
	data: [glam::Vec4; SHADER_POINT_NUM_GROUPS],
}
impl From<Point> for ShaderPoint {
	fn from(point: Point) -> Self {
		let mut data = [glam::Vec4::ZERO; SHADER_POINT_NUM_GROUPS];
		for (i, &f) in <[f32; N + 1]>::from(point).iter().enumerate() {
			data[i / 4][i % 4] = f;
		}
		Self { data }
	}
}
const SHADER_MOTOR_NUM_GROUPS: usize =
	(std::mem::size_of::<Motor>() / std::mem::size_of::<f32>() + 3) / 4;
#[derive(Debug, ShaderType)]
struct ShaderMotor {
	#[align(16)]
	data: [glam::Vec4; SHADER_MOTOR_NUM_GROUPS],
	// #[align(16)]
	// g0: VecN,
	// #[align(16)]
	// g1: VecN,
	// #[align(8)]
	// g2: glam::Vec2,
	// g: [f32; 20],
}
impl From<Motor> for ShaderMotor {
	fn from(motor: Motor) -> Self {
		let mut data = [glam::Vec4::ZERO; SHADER_MOTOR_NUM_GROUPS];
		for (i, &f) in <[f32; 1 << N]>::from(motor).iter().enumerate() {
			let i = MOTOR_INDEX_REMAP[i];
			data[i / 4][i % 4] = f;
		}
		Self { data }
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
	position: Point,
	color: Vec3,
}
#[derive(Debug, ShaderType)]
struct ShaderPointLight {
	#[align(16)]
	position: ShaderPoint,
	#[align(16)]
	color: Vec3,
}
impl From<PointLight> for ShaderPointLight {
	fn from(point_light: PointLight) -> Self {
		Self {
			position: point_light.position.into(),
			color: point_light.color,
		}
	}
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
fn box_verts(half_widths: VecN) -> [Point; 1 << N] {
	std::array::from_fn(|mask| {
		let mut vertex = half_widths;
		for i in 0..N {
			if 1 << i & mask != 0 {
				vertex[i] = -vertex[i];
			}
		}
		let mut coeffs = [1.0f32; N + 1];
		coeffs[..N].copy_from_slice(vertex.as_ref());
		Point::from(coeffs)
	})
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
	motion: Motor,  // "position": M * template "position" * M.reverse == world "position"
	rate: MeetLine, // template "velocity"
	prev_rate: MeetLine, // for velocity-based contact resolution?
	shape: Shape,
	material: Vec3,
}
#[derive(Debug, ShaderType)]
struct ShaderBody {
	#[align(16)]
	motion: ShaderMotor,
	#[align(16)]
	shape: ShaderShape,
	#[align(16)]
	material: Vec3,
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
	key_physics: bool,
	mouse_delta: VecN,
	// mouse_buttons: glam::BVec2, // for the forward/back buttons // TODO
	#[cfg(feature = "four_d")]
	camera_rotate_4d: bool, // false -> rotate planes yz zx xy; true -> rotate planes xw yw zw
	window_size: PhysicalSize<u32>,
	// physics
	time: std::time::Instant,
	speed: f32,
	gravity_accel: MeetLine,
	place_distance: f32,
	penalty_stiffness: f32,
	death_radius_squared: f32,
	// render
	camera: Camera,
	global_light_color: Vec3,
	point_lights: [PointLight; MAX_POINT_LIGHTS],
	num_bodies: usize,
	bodies: [Body; MAX_BODIES],
	floor: Plane,
	floor_color: Vec3,
}
#[derive(Debug, ShaderType)]
struct ShaderState {
	#[align(16)]
	camera: ShaderCamera,
	#[align(16)]
	global_light_color: Vec3,
	#[align(16)]
	floor_color: Vec3,
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
			floor_color: state.floor_color,
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
		if false {
			dbg!(buffer
				.as_ref()
				.chunks_exact(4)
				.map(TryInto::try_into)
				.map(Result::unwrap)
				.map(f32::from_le_bytes)
				.collect::<Vec<_>>());
		}
		Ok(buffer.into_inner())
	}
	fn simulate(&mut self, dt: std::time::Duration) {
		let h = dt.as_secs_f32();
		// update camera
		let t = {
			let mut t = [1.0f32; N + 1];
			t[..N].copy_from_slice(
				((-h * self.speed) * (VecN::from(self.key_plus) - VecN::from(self.key_minus)))
					.as_ref(),
			);
			Translator::from(t)
		};
		let r = {
			let mut r = [0.0f32; 1 << (N - 1)];
			cfg_if! {
				if #[cfg(feature = "four_d")] {
					let offset = if self.camera_rotate_4d { 3 } else { 0 };
				} else {
					let offset = 0;
				}
			}
			r[0 + offset] = self.mouse_delta.x * 1e-3;
			r[1 + offset] = self.mouse_delta.z * 1e-1;
			r[2 + offset] = -self.mouse_delta.y * 1e-3;
			r[(1 << (N - 1)) - 1] = 1.0;
			Rotor::from(r)
		}
		.signum();
		self.mouse_delta = VecN::ZERO;
		self.camera.motion = self
			.camera
			.motion
			.geometric_product(r)
			.geometric_product(t)
			.signum();
		/*
		// this code doesn't work for some reason
		let CAMERA_FLOOR_EPSILON = 0.1f32; // TODO refactor this
		let floor_dist = -self.camera.motion[0] * 2.0;
		if floor_dist < CAMERA_FLOOR_EPSILON {
			self.camera.motion = self.camera.motion.geometric_product(
				{let mut t = [1.0f32; N + 1];
					t[0] = h * 0.5 * floor_dist;
				Translator::from(t)}
			).signum();
		}
		dbg!(&self.camera);
		 */
		if !self.key_physics {
			return;
		}
		// physics simulation
		/*
		Verlet:
		q += h * qdot
		qdot += h * Minv * F(q)
		 */
		// a simplified example can be found here: https://enki.ws/ganja.js/examples/pga_dyn.html
		/*
		let mut next_motions:Vec<Motor> = self.bodies[..self.num_bodies].map(|body|{
			// see table 2.1 https://bivector.net/PGAdyn.pdf
			let d_motion = body.motion.geometric_product(body.rate) * -0.5;
			(body.motion + d_motion * h).signum()
		}).collect();
		 */
		// floor contacts
		let mut contact_forques: Vec<JoinLine> = vec![JoinLine::zero(); self.num_bodies]; // local frame forques
		for (i, body) in self.bodies[..self.num_bodies].iter().enumerate() {
			// dbg!(body.motion.magnitude());
			// we use this cheatsheet: https://bivector.net/3DPGA.pdf
			// and formulas from tables 3 and 4 here https://arxiv.org/abs/2002.04509
			assert!((self.floor.magnitude() - 1.0).abs() < 1e-4);
			let floor = body.motion.reversal().transformation(self.floor).signum();
			let floor_force: Point = {
				let mut data = [0.0f32; N];
				for i in 0..N {
					data[i] = floor[i];
				}
				let n = VecN::from_array(data).normalize();
				let mut data = [0.0f32; N + 1];
				data[..N].copy_from_slice(n.as_ref());
				Point::from(data) * self.penalty_stiffness
			}; // floor * PSS
			assert_eq!(floor_force[N], 0.0);
			assert!((floor.magnitude() - 1.0).abs() < 1e-4);
			// let d_floor_signed_distance = |p: Point| -> f32 {
			// let M = body.motion
			// f = self.floor
			// floor_sdf = (~M f M) V p
			// };
			let floor_signed_distance = |p: Point| -> f32 {
				// "Oriented distance P to p" (Point to plane)
				assert!((floor.squared_magnitude() - 1.0).abs() < 1e-4);
				assert!((p.squared_magnitude() - 1.0).abs() < 1e-4);
				floor.regressive_product(p)
			};
			let project_onto_floor = |p: Point| -> Point {
				// "Line perp to plane p through point P" p dot P
				// "Project point P onto plane p" (p dot P)p
				// TODO make sure no information is lost in the into() conversion to Point
				assert!((floor.squared_magnitude() - 1.0).abs() < 1e-4);
				assert!((p.squared_magnitude() - 1.0).abs() < 1e-4);
				// table 4 https://arxiv.org/pdf/2002.04509 ?
				// p.inner_product(floor).geometric_product(floor).into()
				// table 2 https://bivector.net/PGA4CS.pdf ?
				// floor.inverse().geometric_product(floor.inner_product(p))
				// which one is it??
				p.inner_product(floor).outer_product(floor) // oh
			};
			match body.shape {
				Shape::None => unreachable!(),
				Shape::Sphere(radius) => {
					// floor in template frame
					let floor_center_dist = floor_signed_distance(ORIGIN);
					if floor_center_dist < radius {
						if true {
							// penalty forces
							let d = radius -floor_center_dist;
							let contact_point = project_onto_floor(ORIGIN);
							assert!((contact_point[N] - 1.0).abs() < 1e-4);
							// eq. 2.25 from https://bivector.net/PGAdyn.pdf
							contact_forques[i] += contact_point.regressive_product(floor_force * d);
						} else {
							panic!();
							// velocity-based
							// fake: self.sphere_vels[i].x = -self.sphere_prev_vels[i].x;
							// fake: qnext = self.shader_state.spheres[i].center + h * self.sphere_vels[i];

							// q[t+1] = q[t] + h[t] * qdot[t] // no collision here
							// qdot[t+1] = qdot[t] + h[t] * Minv * (F(q[t+1]) + correction_forque)
							// q[t+2] = q[t+1] + h[t+1] * qdot[t+1] // solve correction_forque such that q[t+2] no collision
							// qdot[t+2] = qdot[t+1] + h[t+1] * Minv * F(q[t+2])
							// we know correction_forque = some multiple of the line normal.
							// next_motions[i] = project_onto_floor(ORIGIN);
							// q[t+2] = q[t+1] + h * (qdot[t] + h * Minv * (F(q[t+1]) + k * plane_normal))
							// ((q[t+2] - q[t+1])/h[t+1] - qdot[t])/h[t] * M - F(q[t+1]) = k * plane_normal
						}
					}
				}
				Shape::Box(half_widths) => {
					// floor is infinite plane, only need to compute vertex-plane collisions
					// N-dimensional box has 2^N vertices
					for vertex in box_verts(half_widths) {
						let floor_vert_dist = floor_signed_distance(vertex);
						if floor_vert_dist < 0.0 {
							let d = -floor_vert_dist;
							let contact_point = project_onto_floor(vertex);
							assert!((contact_point[N] - 1.0).abs() < 1e-4);
							contact_forques[i] += contact_point.regressive_product(floor_force * d);
						}
					}
				}
			}
		}
		for (i, body) in self.bodies[..self.num_bodies].iter_mut().enumerate() {
			// see table 2.1 https://bivector.net/PGAdyn.pdf
			let d_motion = body.motion.geometric_product(body.rate) * -0.5;
			body.motion = (body.motion + d_motion * h).signum();

			let forque: JoinLine = {
				let gravity_forque: JoinLine = body
					.motion
					.reversal()
					.transformation(self.gravity_accel)
					.dual();
				gravity_forque + contact_forques[i]
			};
			let d_rate: MeetLine = {
				let inertia_map_world = |b: MeetLine| -> JoinLine {
					let commutator_product = |p: JoinLine, q: MeetLine| -> JoinLine {
						// hopefully a V (b X c) == (a V b) X c ...
						((p.geometric_product(q) - q.geometric_product(p)) * 0.5).into()
					};
					let commutator_product = |p: Point, q: MeetLine| -> Point {
						dbg!(p,q);
						let q = Motor::zero() + q;
						dbg!(q);
				dbg!(		((p.geometric_product(q) - q.geometric_product(p)) * 0.5).into())
					};
					let world_rate = body.motion.transformation(body.rate);
					// eq. 2.13 https://bivector.net/PGAdyn.pdf#equation.C.1.7
					match body.shape {
						Shape::None => unreachable!(),
						Shape::Sphere(radius) => {
							// it's a sphere so the rotation doesn't matter
							// assume density=1, then density/volume = mass
							let mass: f32 = {
								// https://en.wikipedia.org/wiki/Volume_of_an_n-ball
								match N {
									3 => radius.powi(3) * 4.189,
									4 => radius.powi(4) * 4.935,
									_ => unreachable!(),
								}
							};
							let center = body.motion.transformation(ORIGIN).signum();
							center.regressive_product(commutator_product(center,world_rate)) * mass
							// commutator_product(dbg!(center.regressive_product(center)), world_rate) * mass
						}
						Shape::Box(half_widths) => {
							let mut res = JoinLine::zero();
							// model the rectangle as 2^N corners. (?)
							// mass is evenly distributed to the 2^N vertices
							for vertex in box_verts(half_widths) {
								let vertex = body.motion.transformation(vertex);
								// vertex.regressive_product(commutator_product(vertex,world_rate))
								// res += commutator_product(
								// 	dbg!(vertex.regressive_product(vertex)),
								// 	world_rate,
								// );
								res += dbg!(vertex.regressive_product(commutator_product(vertex, world_rate)));
							}
							res *= 2.0f32.powi(N as i32);
							res
						}
					}
				};
				let inertia_map_body = |b: MeetLine| -> JoinLine {
					// eq. 2.14 https://bivector.net/PGAdyn.pdf
					body.motion.reversal().transformation(
						inertia_map_world(
							body.motion.transformation(
								body.rate
							)
						)
					)
				};
				let inertia_map_body_inv = |p: JoinLine| -> MeetLine {
					// https://bivector.net/PGAdyn.pdf
					// look very carefully at 2.20, 2.21, maybe 2.22 is a good hint
					let inv_rate = inertia_map_body(p.dual()).dual();
					dbg!(inv_rate);
					 dbg!(MeetLine::from(std::array::from_fn(|i|{inv_rate[i].recip()})))
				};
				let res: JoinLine = body.rate.dual(); // TODO replace with inertial map
				// let res: JoinLine = body
				// 	.motion
				// 	.reversal()
				// 	.transformation(inertia_map_world(body.motion.transformation(body.rate)));
				let commutator_product = |p: MeetLine, q: JoinLine| {
					(p.geometric_product(q) - q.geometric_product(p)) * 0.5
				};
				let res = commutator_product(body.rate, res);
				let res = res + forque;
				// TODO this last one should be UNDUAL (which depends on the dimension.....)
				// see eq. 132 here https://geometricalgebra.org/downloads/PGA4CS.pdf
				// grade(forque)=grade(join line)=d-1 i think?, and (d-1)d is always even
				// so we probably (?) don't need a negative sign
				// let res = res.dual(); // TODO replace with inverse inertial map
				// let res = inertia_map_body_inv(res.into());
				let res = res.dual();
				res.into()
				// TODO why dual? re-read eq. 2.25 from https://bivector.net/PGAdyn.pdf
			};
			body.prev_rate = body.rate;
			body.rate += d_rate * h;
		}
		// delete bodies that are too far
		let mut new_len = 0;
		for i in 0..self.num_bodies {
			let position =
				VecN::from_array(std::array::from_fn(|e| -2.0 * self.bodies[i].motion[e]));
			if position.length_squared() < self.death_radius_squared {
				self.bodies[new_len] = self.bodies[i].clone();
				new_len += 1;
			}
		}
		self.num_bodies = new_len;
		self.bodies[new_len].shape = Shape::None;
	}
	fn try_place_body(&mut self) {
		if self.num_bodies == MAX_BODIES {
			return;
		}
		let motion = self.camera.motion.geometric_product(
			Motor::one() - BASIS[0].outer_product(BASIS[2]) * (0.5 * self.place_distance),
		);
		let rand_around_1 = || (rand::random::<f32>() - 0.5) + 1.0;
		self.bodies[self.num_bodies] = Body {
			motion,
			rate: MeetLine::zero(),
			prev_rate: MeetLine::zero(),
			shape: if rand::random::<f32>() < 0.25 {
				Shape::Sphere(rand_around_1())
			} else {
				Shape::Box(VecN::from_array(std::array::from_fn(|_| rand_around_1())))
			},
			material: Vec3::new(rand::random(), rand::random(), rand::random()),
		};
		dbg!(&self.bodies[self.num_bodies]);
		self.num_bodies += 1;
	}
}

async fn run(event_loop: EventLoop<()>, window: Window) {
	let mut app_state = AppState {
		// controls
		key_plus: BVecN::FALSE,
		key_minus: BVecN::FALSE,
		key_physics: false,
		mouse_delta: VecN::ZERO,
		#[cfg(feature = "four_d")]
		camera_rotate_4d: false,
		window_size: window.inner_size(),
		// physics
		time: std::time::Instant::now(),
		speed: 1.0,
		gravity_accel: dbg!(BASIS[0].outer_product(BASIS[1]) * -10.0),
		place_distance: 4.0,
		penalty_stiffness: 1e3, // TODO experiment
		death_radius_squared: 128.0 * 128.0,
		// render
		camera: dbg!(Camera {
			motion: dbg!(Motor::one() - BASIS[0].outer_product(BASIS[1]) * 0.5),
			up: BASIS[1].dual(),
			forward: BASIS[2].dual(), // TODO should `forward` be a hyperplane? (dual of point)
			left: BASIS[3].dual(),
			// we don't need an additional camera param in 4D because the camera window is always 2d
		}),
		global_light_color: Vec3::ONE * 0.25,
		point_lights: [PointLight {
			color: Vec3::ONE,
			position: dbg!(ORIGIN + BASIS[1].dual() * 128.0),
		}; MAX_POINT_LIGHTS],
		floor: BASIS[1],
		floor_color: Vec3::ONE * 0.1,
		num_bodies: 0,
		bodies: std::array::from_fn(|_| Body {
			motion: Motor::zero(), // zero motor is degenerate, make sure body is treated as null
			rate: MeetLine::zero(),
			prev_rate: MeetLine::zero(),
			shape: Shape::None,
			material: Vec3::ONE,
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
			compatible_surface: Some(&surface),
		})
		.await
		.expect("Failed to find an appropriate adapter");

	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::downlevel_webgl2_defaults()
					.using_resolution(adapter.limits()),
			},
			None,
		)
		.await
		.expect("Failed to create device");

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
	//window.set_fullscreen(Some(Fullscreen::Borderless(None)));
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
						// dbg!(button, state);
						let pressed = state == ElementState::Pressed;
						match button {
							0 => {
								// mouse left
								if pressed {
									app_state.try_place_body();
								}
							}
							1 => {
								// mouse right
								#[cfg(feature = "four_d")]
								{
									app_state.camera_rotate_4d = pressed;
								}
							}
							2 => { // mouse middle
							}
							3 => {
								// mouse back
								if pressed {
									app_state.mouse_delta.z += 1.0;
								}
							}
							4 => {
								// mouse front
								if pressed {
									app_state.mouse_delta.z -= 1.0;
								}
							}
							_ => {
								dbg!(button);
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
								#[cfg(feature = "four_d")]
								KeyCode::KeyQ => app_state.key_plus.w = pressed,
								#[cfg(feature = "four_d")]
								KeyCode::KeyE => app_state.key_minus.w = pressed,
								KeyCode::Enter => app_state.key_physics = pressed,
								_ => {}
							}
						}
					}
					_ => {}
				},
				Event::WindowEvent { event, .. } => match event {
					WindowEvent::Resized(new_size) => {
						// Reconfigure the surface with the new size
						config.width = new_size.width;
						config.height = new_size.height;
						while config.width * config.height > 1048576 {
							config.width >>= 1;
							config.height >>= 1;
						}
						config.width = config.width.max(1);
						config.height = config.height.max(1);
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
