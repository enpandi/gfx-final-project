use encase::ShaderType;
use std::borrow::Cow;
use winit::{
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window},
};
use winit::event::RawKeyEvent;
use geometric_algebra::ppga3d;

#[derive(Debug, ShaderType)]
struct ShaderState {
    pub cam_pos: glam::Vec3,
}

impl ShaderState {
    fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(self)?;
        Ok(buffer.into_inner())
    }
}

#[derive(Debug)]
struct AppState {
    time: std::time::Instant,
    delta_time: Option<std::time::Duration>,
    speed: f32,
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    shader_state: ShaderState,
}

impl AppState {
    fn update_time(&mut self) {
        let time = std::time::Instant::now();
        self.delta_time = Some(time - self.time);
        self.time = time;
    }
    fn apply_move(&mut self) {
        let vel = self.speed
            * self
                .delta_time
                .expect("apply_move should be called after update_time")
                .as_secs_f32();
        if self.move_forward {
            self.shader_state.cam_pos.z -= vel;
        }
        if self.move_backward {
            self.shader_state.cam_pos.z += vel;
        }
        if self.move_left {
            self.shader_state.cam_pos.x -= vel;
        }
        if self.move_right {
            self.shader_state.cam_pos.x += vel;
        }
        if self.move_up {
            self.shader_state.cam_pos.y += vel;
        }
        if self.move_down {
            self.shader_state.cam_pos.y -= vel;
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            time: std::time::Instant::now(),
            delta_time: None,
            speed: 8.0,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            shader_state: ShaderState {
                cam_pos: glam::Vec3::ZERO,
            },
        }
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut app_state = AppState::default();

    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

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
        size: app_state.shader_state.size().get(),
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
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    // window .set_cursor_grab(CursorGrabMode::Confined) .expect("set_cursor_grab failed");
    // event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

            match event {
                Event::DeviceEvent { event, .. } => match event {
                    DeviceEvent::MouseMotion { delta : (x, y)} => {
                        dbg!(x, y);
                    }
                    DeviceEvent::MouseWheel { delta } => {
                        dbg!(delta);
                    }
                    DeviceEvent::Key(RawKeyEvent{physical_key, state}) => {
                        if let PhysicalKey::Code(key_code) = physical_key {
                            let pressed = state == ElementState::Pressed;
                            match key_code {
                                KeyCode::KeyW => app_state.move_forward = pressed,
                                KeyCode::KeyA => app_state.move_left = pressed,
                                KeyCode::KeyS => app_state.move_backward = pressed,
                                KeyCode::KeyD => app_state.move_right = pressed,
                                KeyCode::ShiftLeft => app_state.move_down = pressed,
                                KeyCode::Space => app_state.move_up = pressed,
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        app_state.update_time();
                        app_state.apply_move();
                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        queue.write_buffer(
                            &uniform_buffer,
                            0,
                            &app_state.shader_state.as_wgsl_bytes().expect(
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
