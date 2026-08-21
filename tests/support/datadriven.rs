//! A minimal data-driven test runner, modeled on the CockroachDB datadriven
//! format.
//!
//! Corpus files under `tests/datadriven/` are sequences of blocks:
//!
//! ```text
//! # Comment lines describing the block.
//! command arg=value flag
//! input line
//! ...
//! ----
//! expected output line
//! ...
//!
//! ```
//!
//! A block is an optional comment group, one command line, zero or more input
//! lines, a `----` separator, and the expected output. Blocks are separated by
//! exactly one blank line; expected output never contains a blank line, and a
//! command whose output is empty has its `----` followed immediately by the
//! separating blank line. Command arguments are whitespace-separated
//! `key=value` pairs or bare flags; values never contain spaces (vectors are
//! written `[1,2,-3]` without inner spaces).
//!
//! The runner executes every directive in file order and diffs the actual
//! output against the expected text. Setting `KTANN_REWRITE=1` regenerates the
//! corpus with actual outputs instead of failing, preserving comments,
//! command lines, and inputs verbatim; a rewrite is a code-review checkpoint,
//! not a way to silence a failure.

use std::fmt;
use std::path::{Path, PathBuf};

/// The environment variable that switches the runner into rewrite mode.
pub const REWRITE_ENV: &str = "KTANN_REWRITE";

/// One parsed corpus block: one directive invocation with its expectations.
#[derive(Clone, Debug)]
pub struct Directive {
    /// The comment group above the command, preserved verbatim for rewrite.
    pub comments: String,
    /// The command line exactly as written (command name and arguments).
    pub raw_header: String,
    /// Arguments in written order: `key=value` pairs and bare flags.
    pub args: Vec<(String, Option<String>)>,
    /// Input lines between the command line and the `----` separator.
    pub input: Vec<String>,
    /// The expected output between `----` and the terminating blank line.
    pub expected: String,
    /// The 1-based line number of the command line in the source file.
    pub line: usize,
}

impl Directive {
    /// The command name: the first token of the raw command line.
    #[must_use]
    pub fn command(&self) -> &str {
        self.raw_header
            .split_whitespace()
            .next()
            .expect("a parsed directive has a command")
    }

    /// Returns the value of a `key=value` argument, if present.
    #[must_use]
    pub fn arg(&self, key: &str) -> Option<&str> {
        self.args
            .iter()
            .find(|(name, _)| name == key)
            .and_then(|(_, value)| value.as_deref())
    }

    /// Returns whether a bare flag argument is present.
    #[must_use]
    pub fn flag(&self, key: &str) -> bool {
        self.args
            .iter()
            .any(|(name, value)| name == key && value.is_none())
    }

    /// Returns the value of a required argument, or fails with context.
    #[must_use]
    pub fn require(&self, key: &str) -> &str {
        self.arg(key).unwrap_or_else(|| {
            panic!(
                "directive `{}` at line {} requires `{key}=`",
                self.raw_header, self.line
            )
        })
    }

    /// Parses a non-negative integer argument with a default.
    #[must_use]
    pub fn arg_usize(&self, key: &str, default: usize) -> usize {
        match self.arg(key) {
            Some(value) => value.parse().unwrap_or_else(|_| {
                panic!(
                    "directive `{}` at line {}: `{key}=` must be a non-negative integer, got `{value}`",
                    self.raw_header, self.line
                )
            }),
            None => default,
        }
    }

    /// Parses a non-negative 64-bit integer argument with a default.
    #[must_use]
    pub fn arg_u64(&self, key: &str, default: u64) -> u64 {
        match self.arg(key) {
            Some(value) => value.parse().unwrap_or_else(|_| {
                panic!(
                    "directive `{}` at line {}: `{key}=` must be a non-negative integer, got `{value}`",
                    self.raw_header, self.line
                )
            }),
            None => default,
        }
    }

    /// Returns every value of a repeated `key=value` argument.
    pub fn args_of<'a>(&'a self, key: &'a str) -> impl Iterator<Item = &'a str> {
        self.args
            .iter()
            .filter(move |(name, _)| name == key)
            .filter_map(|(_, value)| value.as_deref())
    }
}

/// Parses one corpus file into its directive sequence.
///
/// Malformed files are a hard test-authoring error, hence the panic with the
/// offending line number.
pub fn parse(path: &Path, text: &str) -> Vec<Directive> {
    let mut directives = Vec::new();
    let mut comments = String::new();
    let mut lines = text.lines().enumerate().peekable();

    while let Some((_, line)) = lines.peek() {
        if line.is_empty() {
            lines.next();
            continue;
        }
        if line.starts_with('#') {
            comments.push_str(line);
            comments.push('\n');
            lines.next();
            continue;
        }
        if line.trim() == "----" {
            panic!("{}: stray `----` separator", path.display());
        }

        // The command line.
        let (number, raw_header) = lines.next().expect("peeked line must exist");
        let mut tokens = raw_header.split_whitespace();
        if tokens.next().is_none() {
            panic!("{}:{}: empty command line", path.display(), number + 1);
        }
        let args = tokens
            .map(|token| match token.split_once('=') {
                Some((key, value)) => (key.to_string(), Some(value.to_string())),
                None => (token.to_string(), None),
            })
            .collect();

        // Input lines up to the separator.
        let mut input = Vec::new();
        let mut separated = false;
        for (_, line) in lines.by_ref() {
            if line.trim() == "----" {
                separated = true;
                break;
            }
            if line.is_empty() || line.starts_with('#') {
                panic!(
                    "{}: directive `{raw_header}` is missing its `----` separator",
                    path.display()
                );
            }
            input.push(line.to_string());
        }
        if !separated {
            panic!(
                "{}: directive `{raw_header}` is missing its `----` separator",
                path.display()
            );
        }

        // Expected output up to the terminating blank line or end of file.
        let mut expected = String::new();
        for (_, line) in lines.by_ref() {
            if line.is_empty() {
                break;
            }
            if line.starts_with('#') || line.trim() == "----" {
                panic!(
                    "{}: directive `{raw_header}` output must end with a blank line",
                    path.display()
                );
            }
            expected.push_str(line);
            expected.push('\n');
        }

        directives.push(Directive {
            comments: std::mem::take(&mut comments),
            raw_header: raw_header.to_string(),
            args,
            input,
            expected,
            line: number + 1,
        });
    }
    directives
}

/// Renders the corpus back out with `outputs` substituted as expected output.
///
/// `outputs` must have exactly one entry per directive.
pub fn render(directives: &[Directive], outputs: &[String]) -> String {
    assert_eq!(directives.len(), outputs.len(), "one output per directive");
    let mut text = String::new();
    for (directive, output) in directives.iter().zip(outputs) {
        text.push_str(&directive.comments);
        text.push_str(&directive.raw_header);
        text.push('\n');
        for line in &directive.input {
            text.push_str(line);
            text.push('\n');
        }
        text.push_str("----\n");
        text.push_str(output);
        text.push('\n');
    }
    text
}

/// The failure of one directive's output comparison.
#[derive(Debug)]
pub struct Mismatch {
    /// The corpus file.
    pub path: PathBuf,
    /// The 1-based command line number.
    pub line: usize,
    /// The command line as written.
    pub raw_header: String,
    /// The recorded expected output.
    pub expected: String,
    /// The freshly computed output.
    pub actual: String,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{} (directive `{}`)\n--- expected ---\n{}\n--- actual ---\n{}",
            self.path.display(),
            self.line,
            self.raw_header,
            self.expected,
            self.actual
        )
    }
}

/// Renders a corpus Record ID (corpus IDs are UTF-8).
#[must_use]
pub fn show_id(id: &bytes::Bytes) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(id)
}

/// Whether the runner rewrites the corpus instead of failing on a mismatch.
#[must_use]
pub fn rewrite_enabled() -> bool {
    std::env::var(REWRITE_ENV).is_ok_and(|value| value == "1")
}

/// Lists the corpus files of one directory, sorted for deterministic order.
pub fn corpus_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
        .map(|entry| entry.expect("corpus entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "kddt"))
        .collect();
    files.sort();
    files
}
