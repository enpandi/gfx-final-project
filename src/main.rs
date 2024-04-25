use core::cmp::Ord;
use encase::{ShaderSize, ShaderType};
use geometric_algebra::{
	ppga3d, GeometricProduct, Magnitude, One, Signum, SquaredMagnitude, Transformation,
};
use std::borrow::Cow;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::RawKeyEvent;
use winit::{
	event::{DeviceEvent, ElementState, Event, WindowEvent},
	event_loop::EventLoop,
	keyboard::{KeyCode, PhysicalKey},
	window::Window,
};

#[derive(Debug, ShaderType)]
struct ShaderState {
	pub cam_pos: glam::Vec3,
	pub cam_up: glam::Vec3,
	pub cam_forward: glam::Vec3,
	pub cam_left: glam::Vec3,
}

impl ShaderState {
	fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
		let mut buffer = encase::UniformBuffer::new(Vec::new());
		buffer
			.write(self)
			.expect("encase::UniformBuffer::write error");
		Ok(buffer.into_inner())
	}
}

#[derive(Debug)]
struct AppState {
	time: std::time::Instant,
	speed: f32,
	cam: ppga3d::Motor,
	key_plus: glam::BVec3,
	key_minus: glam::BVec3,
	mouse_delta: glam::Vec2,
	window_size: PhysicalSize<u32>,
	fov: f32,
}

impl AppState {
	fn simulate(&mut self) {
		/*
		q[i+1] = q[i] + h qdot[i]
		qdot[i+1] = qdot[i] - h * Minv * dV(q[i+1])
		===
		q += h * qdot
		qdot += h * Minv * F(q)
		 */
		let h = {
			let time = std::time::Instant::now();
			let h = (time - self.time).as_secs_f32();
			self.time = time;
			h
		};
		let t = {
			let t = -h
				* self.speed * (glam::Vec3::from(self.key_plus)
				- glam::Vec3::from(self.key_minus));
			ppga3d::Translator::new(1.0, t.x, t.y, t.z)
		};
		let r = ppga3d::Rotor::new(
			1.0,
			self.mouse_delta.x * 1e-3,
			0.0,
			-self.mouse_delta.y * 1e-3,
		)
		.signum();
		self.mouse_delta = glam::Vec2::ZERO;
		self.cam = (self.cam.geometric_product(r).geometric_product(t)).signum();
	}
	fn shader_state(&self) -> ShaderState {
		let cam_pos = self
			.cam
			.transformation(ppga3d::Point::new(1.0, 0.0, 0.0, 0.0));
		let cam_up = self.cam.transformation(ppga3d::Point::new(
			0.0,
			self.window_size.height as f32,
			0.0,
			0.0,
		));
		let cam_forward = self.cam.transformation(ppga3d::Point::new(
			0.0,
			0.0,
			((self.window_size.width * self.window_size.width
				+ self.window_size.height * self.window_size.height) as f32)
				.sqrt() / (0.5 * self.fov).tan(),
			0.0,
		));
		let cam_left = self.cam.transformation(ppga3d::Point::new(
			0.0,
			0.0,
			0.0,
			self.window_size.width as f32,
		));
		ShaderState {
			cam_pos: glam::Vec3::new(cam_pos[1], cam_pos[2], cam_pos[3]),
			cam_up: glam::Vec3::new(cam_up[1], cam_up[2], cam_up[3]),
			cam_forward: glam::Vec3::new(cam_forward[1], cam_forward[2], cam_forward[3]),
			cam_left: glam::Vec3::new(cam_left[1], cam_left[2], cam_left[3]),
		}
	}
}

async fn run(event_loop: EventLoop<()>, window: Window) {
	let mut app_state = AppState {
		time: std::time::Instant::now(),
		speed: 1.0,
		cam: ppga3d::Motor::one(),
		key_plus: glam::BVec3::FALSE,
		key_minus: glam::BVec3::FALSE,
		mouse_delta: glam::Vec2::ZERO,
		window_size: window.inner_size(),
		fov: std::f32::consts::FRAC_PI_2,
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
		source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
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
	surface.configure(&device, &config);

	let window = &window;
	// window .set_cursor_grab(CursorGrabMode::Confined) .expect("set_cursor_grab failed");
	// event_loop.set_control_flow(ControlFlow::Poll);
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
						dbg!(delta);
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
						surface.configure(&device, &config);
						// On macos the window needs to be redrawn manually after resizing
						window.request_redraw();
					}
					WindowEvent::RedrawRequested => {
						app_state.simulate();
						let frame = surface
							.get_current_texture()
							.expect("Failed to acquire next swap chain texture");
						let view = frame
							.texture
							.create_view(&wgpu::TextureViewDescriptor::default());
						queue.write_buffer(
							&uniform_buffer,
							0,
							&app_state.shader_state().as_wgsl_bytes().expect(
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
	let mut builder = winit::window::WindowBuilder::new();
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
