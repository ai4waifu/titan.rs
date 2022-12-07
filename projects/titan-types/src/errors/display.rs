use super::*;

impl Error for TitanError {}

impl Debug for TitanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.kind, f)
    }
}

impl Display for TitanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.kind, f)
    }
}

impl Display for TitanErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TitanErrorKind::UnknownError => {
                write!(f, "UnknownError")
            }
        }
    }
}
