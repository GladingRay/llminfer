use minijinja::{context, Environment, Error, ErrorKind};
use std::ffi::c_void;

#[repr(C)]
struct Input {
    data: *const u8,
    size: usize,
}

impl Input {
    fn as_str(&self) -> Result<&str, String> {
        if self.size == 0 {
            return Ok("");
        }
        std::str::from_utf8(unsafe { std::slice::from_raw_parts(self.data, self.size) })
            .map_err(|err| format!("Chat template input must be UTF-8: {err}"))
    }
}

#[repr(C)]
struct ChatInput {
    source: Input,
    prompt: Input,
    bos_token: Input,
    eos_token: Input,
    enable_thinking: u8,
}

fn render(input: &ChatInput) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.render_str(
        input.source.as_str()?,
        context! {
            messages => [context! { role => "user", content => input.prompt.as_str()? }],
            tools => (),
            documents => (),
            bos_token => input.bos_token.as_str()?,
            eos_token => input.eos_token.as_str()?,
            add_generation_prompt => true,
            enable_thinking => input.enable_thinking != 0,
        },
    )
    .map_err(|err| format!("{err}"))
}

#[no_mangle]
extern "C" fn llmi_render_chat_template(
    input: &ChatInput,
    context: *mut c_void,
    output: extern "C" fn(*mut c_void, *const u8, usize),
) -> i32 {
    let (text, status) = match render(input) {
        Ok(text) => (text, 0),
        Err(error) => (error, 1),
    };
    output(context, text.as_ptr(), text.len());
    status
}
