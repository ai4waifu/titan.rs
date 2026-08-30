use super::*;

impl From<TitanErrorKind> for TitanError {
    fn from(value: TitanErrorKind) -> Self {
        TitanError::from_kind(value)
    }
}
