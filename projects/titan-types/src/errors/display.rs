use super::*;

impl Error for TitanError {}

impl Debug for TitanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TitanError")
            .field("kind", &self.kind)
            .field("detail", &self.detail)
            .finish()
    }
}

impl Display for TitanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(detail) if !detail.is_empty() => {
                write!(f, "{}: {detail}", self.kind.as_str())
            }
            _ => Display::fmt(&*self.kind, f),
        }
    }
}

impl Display for TitanErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
