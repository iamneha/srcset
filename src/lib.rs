
use std::collections::LinkedList;
use std::iter::Peekable;

type StaticCharVec = &'static [char];
pub static HTML_SPACE_CHARACTERS: StaticCharVec = &[' ', '\t', '\n', '\u{c}', '\r'];

fn is_ascii_digit(c: &char) -> bool {
    match *c {
        '0'...'9' => true,
        _ => false,
    }
}

/// Reads an exponent from an iterator over chars, for example `e100`.
pub fn read_exponent<I: Iterator<Item=char>>(mut iter: Peekable<I>) -> Option<i32> {
    match iter.peek() {
        Some(c) if is_exponent_char(*c) => (),
        _ => return None,
    }
    iter.next();

    match iter.peek() {
        None => None,
        Some(&'-') => {
            iter.next();
            read_numbers(iter).0.map(|exp| -exp.to_i32().unwrap_or(0))
        }
        Some(&'+') => {
            iter.next();
            read_numbers(iter).0.map(|exp| exp.to_i32().unwrap_or(0))
        }
        Some(_) => read_numbers(iter).0.map(|exp| exp.to_i32().unwrap_or(0))
    }
}

fn is_decimal_point(c: char) -> bool {
    c == '.'
}

fn is_exponent_char(c: char) -> bool {
    match c {
        'e' | 'E' => true,
        _ => false,
    }
}

trait NumCast {
    fn to_i32(&self) -> Option<i32>;
    fn to_f64(&self) -> Option<f64>;
    fn to_u32(&self) -> Option<u32>;
}
impl NumCast for i64 {
    fn to_i32(&self) -> Option<i32> {
        Some(*self as i32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn to_u32(&self) -> Option<u32> {
        Some(*self as u32)
    }
}

pub fn read_fraction<I: Iterator<Item=char>>(mut iter: Peekable<I>,
                                             mut divisor: f64,
                                             value: f64) -> (f64, usize) {
    match iter.peek() {
        Some(c) if is_decimal_point(*c) => (),
        _ => return (value, 0),
    }
    iter.next();

    iter.take_while(is_ascii_digit).map(|d|
        d as i64 - '0' as i64
    ).fold((value, 1), |accumulator, d| {
        divisor *= 10f64;
        (accumulator.0 + d as f64 / divisor, accumulator.1 + 1)
    })
}

/// Read a set of ascii digits and read them into a number.
pub fn read_numbers<I: Iterator<Item=char>>(mut iter: Peekable<I>) -> (Option<i64>, usize) {
    match iter.peek() {
        Some(c) if is_ascii_digit(c) => (),
        _ => return (None, 0),
    }

    iter.take_while(is_ascii_digit).map(|d| {
        d as i64 - '0' as i64
    }).fold((Some(0i64), 0), |accumulator, d| {
        let digits = accumulator.0.and_then(|accumulator| {
            accumulator.checked_mul(10)
        }).and_then(|accumulator| {
            accumulator.checked_add(d)
        });
        (digits, accumulator.1 + 1)
    })
}

#[derive(Clone)]
enum Error {
    Yes,
    No,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum ParseState {
    InDescriptor,
    InParens,
    AfterDescriptor,
}

#[derive(Debug)]
#[derive(PartialEq)]
pub struct ImageSource {
    pub url: String,
    pub descriptor: Descriptor,
}
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Descriptor {
    pub wid: Option<u32>,
    pub den: Option<f64>,
}

pub fn collect_sequence_characters<F>(s: &str, predicate: F) -> (&str, &str)
    where F: Fn(&char) -> bool
    {
        for (i, ch) in s.chars().enumerate() {
            if !predicate(&ch) {
                return (&s[0..i], &s[i..])
            }
        }
    return (s, "");
    }

pub fn parse_a_srcset_attribute(input: String) -> Vec<ImageSource> {
    let mut start = 0;
    let mut candidates: Vec<ImageSource> = Vec::new();
    while start < input.len() {
        let position = &input[start..];
        let(spaces, position) = collect_sequence_characters(position, |c| *c ==',' || char::is_whitespace(*c));
        let x = spaces.find(',');
        match x {
            Some(val) => println!("Parse Error"),
            None => println!("No commas\n"),
        }
        // add the counts of spaces that we collect to advance the start index
        let space_len = spaces.char_indices().count();
        start += space_len;
        //Returns and breaks out of the loop
        if position.is_empty() {
            return candidates;
        }
        let(url, spaces) = collect_sequence_characters(position, |c| !char::is_whitespace(*c));
        // add the counts of urls that we parse to advance the start index
        start += url.chars().count();
        let comma_count = url.chars().rev().take_while(|c| *c == ',').count();
        let url: String = url.chars().take(url.chars().count() - comma_count).collect();
        if comma_count > 1 {
            println!("Parse Error (trailing commas)")
        }
        // add 1 to start index, for the comma
        start += 1;
        let(space, position) = collect_sequence_characters(spaces, |c| char::is_whitespace(*c));
        let space_len = space.len();
        start += space_len;
        let mut descriptors = LinkedList::<String>::new();
        let mut current_descriptor = String::new();
        let mut state = ParseState::InDescriptor;
        let mut char_stream = position.chars().enumerate();
        let mut buffered: Option<(usize, char)> = None;
        loop {
            let nextChar = buffered.take().or_else(|| char_stream.next());
            if let Some((i, _)) = nextChar {
                start += 1;
            }
            match state {
                ParseState::InDescriptor => {
                    match nextChar {
                        Some((idx, c @ ' ')) => {
                            if !current_descriptor.is_empty() {
                                descriptors.push_back(current_descriptor.clone());
                                current_descriptor = String::new();
                                state = ParseState::AfterDescriptor;
                            }
                            continue;
                        }
                        Some((idx, c @ ',')) => {
                            position.chars().enumerate();
                            if !current_descriptor.is_empty() {
                                descriptors.push_back(current_descriptor.clone());
                            }
                            break;
                        }
                        Some((idx, c @ '(')) => {
                            current_descriptor.push(c);
                            state = ParseState::InParens;
                            continue;
                        }
                        Some((_, c)) => {
                            current_descriptor.push(c);
                            continue;
                        }
                        None => {
                            if !current_descriptor.is_empty() {
                                descriptors.push_back(current_descriptor.clone());
                            }
                            break;
                        }
                    }
                }
                ParseState::InParens => {
                    match nextChar {
                        Some((idx, c @ ')')) => {
                            current_descriptor.push(c);
                            state = ParseState::InDescriptor;
                            continue;
                        }
                        Some((_, c)) => {
                            current_descriptor.push(c);
                            continue;
                        }
                        None => {
                            if !current_descriptor.is_empty() {
                                descriptors.push_back(current_descriptor.clone());
                            }
                            break;
                        }
                    }
                }
                ParseState::AfterDescriptor => {
                    match nextChar {
                        Some((idx, ' ')) => {
                            state = ParseState::AfterDescriptor;
                            continue;
                        }
                        Some((idx, c)) => {
                            state = ParseState::InDescriptor;
                            buffered = Some((idx, c));
                            continue;
                        }
                        None => {
                            if !current_descriptor.is_empty() {
                                descriptors.push_back(current_descriptor.clone());
                            }
                            break;
                        }
                    }
                }
            }
        }
        let mut error = false;
        let mut width: Option<u32> = None;
        let mut density: Option<f64> = None;
        let mut future_compat_h: Option<u32> = None;
        for descriptor in descriptors {
            let char_iter = descriptor.chars();
            let (digits, remaining) = collect_sequence_characters(&descriptor, is_ascii_digit);
            let valid_non_negative_integer = parse_unsigned_integer(digits.chars());
            let has_w = remaining == "w";
            let valid_floating_point = parse_double(digits);
            let has_x = remaining == "x";
            let has_h = remaining == "h";
            if valid_non_negative_integer.is_ok() && has_w {
                //not support sizes attribute
                error = width.is_some() && density.is_some();
                let result = parse_unsigned_integer(char_iter.clone());
                error = result.is_err();
                if let Ok(w) = result {
                    width = Some(w);
                }
            } else if valid_floating_point.is_ok() && has_x {
                if width.is_some() && density.is_some() && future_compat_h.is_some() {
                    error = true;
                }
                let result = parse_double(char_iter.as_str());
                error = result.is_err();
                if let Ok(x) = result {
                    density = Some(x);
                }
            } else if valid_non_negative_integer.is_ok() && has_h {
                if density.is_some() && future_compat_h.is_some() {
                    error = true;
                }
                let result = parse_unsigned_integer(char_iter.clone());
                error = result.is_err();
                if let Ok(h) = result {
                    future_compat_h = Some(h);
                }
            } else {
                error = true;
            }
        }
        if future_compat_h.is_some() && width.is_none() {
            error = true;
        }
        if !error {
                let descriptor = Descriptor { wid: width, den: density };
                let imageSource = ImageSource { url: url, descriptor: descriptor };
                candidates.push(imageSource);
            
        }
    }
    candidates
}

/// Shared implementation to parse an integer according to
/// <https://html.spec.whatwg.org/multipage/#rules-for-parsing-integers> or
/// <https://html.spec.whatwg.org/multipage/#rules-for-parsing-non-negative-integers>
fn do_parse_integer<T: Iterator<Item=char>>(input: T) -> Result<i64, ()> {
    let mut input = input.skip_while(|c| {
        HTML_SPACE_CHARACTERS.iter().any(|s| s == c)
    }).peekable();

    let sign = match input.peek() {
        None => return Err(()),
        Some(&'-') => {
            input.next();
            -1
        },
        Some(&'+') => {
            input.next();
            1
        },
        Some(_) => 1,
    };

    let (value, _) = read_numbers(input);

    value.and_then(|value| value.checked_mul(sign)).ok_or(())
}

/// Parse an integer according to
/// <https://html.spec.whatwg.org/multipage/#rules-for-parsing-non-negative-integers>
pub fn parse_unsigned_integer<T: Iterator<Item=char>>(input: T) -> Result<u32, ()> {
    do_parse_integer(input).and_then(|result| {
        result.to_u32().ok_or(())
    })
}

/// Parse a floating-point number according to
/// <https://html.spec.whatwg.org/multipage/#rules-for-parsing-floating-point-number-values>
pub fn parse_double(string: &str) -> Result<f64, ()> {
    let trimmed = string.trim_matches(HTML_SPACE_CHARACTERS);
    let mut input = trimmed.chars().peekable();

    let (value, divisor, chars_skipped) = match input.peek() {
        None => return Err(()),
        Some(&'-') => {
            input.next();
            (-1f64, -1f64, 1)
        }
        Some(&'+') => {
            input.next();
            (1f64, 1f64, 1)
        }
        _ => (1f64, 1f64, 0)
    };

    let (value, value_digits) = if let Some(&'.') = input.peek() {
        (0f64, 0)
    } else {
        let (read_val, read_digits) = read_numbers(input);
        (value * read_val.and_then(|result| result.to_f64()).unwrap_or(1f64), read_digits)
    };

    let input = trimmed.chars().skip(value_digits + chars_skipped).peekable();

    let (mut value, fraction_digits) = read_fraction(input, divisor, value);

    let input = trimmed.chars().skip(value_digits + chars_skipped + fraction_digits).peekable();

    if let Some(exp) = read_exponent(input) {
        value *= 10f64.powi(exp)
    };

    Ok(value)
}

#[test]
fn no_value() {
    //println!("{:?}", parse_a_srcset_attribute(String::new()));
    let v = Vec::new();
    assert_eq!(parse_a_srcset_attribute(String::new()), v);
}

#[test]
fn one_value() {
    //println!("test: {:?}", parse_a_srcset_attribute(String::from("elva-fairy-320w.jpg 320w, elva-fairy-480w.jpg 480w")));
    let d = Descriptor { wid: Some(320), den: None };
    let v = ImageSource {url: "elva-fairy-320w.jpg".to_string(), descriptor: d};
    let mut sources = Vec::new();
    sources.push(v);
    assert_eq!(parse_a_srcset_attribute(String::from("elva-fairy-320w.jpg 320w")), sources);
}
