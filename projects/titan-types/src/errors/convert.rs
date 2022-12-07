use super::*;

impl From<TitanErrorKind> for TitanError {
    fn from(value: TitanErrorKind) -> Self {
        Self { kind: Box::new(value) }
    }
}
