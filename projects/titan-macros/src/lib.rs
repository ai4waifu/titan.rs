#![warn(missing_docs)]
//! Compile-time validation and stable metadata markers for Titan declarations.
//!
//! These macros deliberately validate declarations only. They do not create a
//! model runtime, communication plan, kernel IR, or a second expression IR.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Error, Expr, ExprLit, ExprPath, Item, ItemFn, ItemStruct, Lit, Meta, Result, Token, parse::Parser, parse_macro_input,
    punctuated::Punctuated,
};

#[derive(Clone, Copy)]
enum Declaration {
    Neural,
    Parameters,
    Kernel,
    Distributed,
}

impl Declaration {
    const fn name(self) -> &'static str {
        match self {
            Self::Neural => "neural",
            Self::Parameters => "parameters",
            Self::Kernel => "kernel",
            Self::Distributed => "distributed",
        }
    }

    const fn accepts(self, item: &Item) -> bool {
        match self {
            Self::Neural | Self::Distributed => matches!(item, Item::Fn(_) | Item::Struct(_)),
            Self::Parameters => matches!(item, Item::Struct(_)),
            Self::Kernel => matches!(item, Item::Fn(_)),
        }
    }

    const fn expected_item(self) -> &'static str {
        match self {
            Self::Neural | Self::Distributed => "函数或结构体",
            Self::Parameters => "结构体",
            Self::Kernel => "函数",
        }
    }
}

fn ident(item: &Item) -> &syn::Ident {
    match item {
        Item::Fn(ItemFn { sig, .. }) => &sig.ident,
        Item::Struct(ItemStruct { ident, .. }) => ident,
        _ => unreachable!("items are validated before their identifier is requested"),
    }
}

fn no_arguments(attributes: proc_macro2::TokenStream, declaration: Declaration) -> Result<()> {
    if attributes.is_empty() {
        Ok(())
    }
    else {
        Err(Error::new_spanned(attributes, format!("#[{}] 不接受属性参数", declaration.name())))
    }
}

fn metas(attributes: proc_macro2::TokenStream) -> Result<Punctuated<Meta, Token![,]>> {
    Punctuated::<Meta, Token![,]>::parse_terminated.parse2(attributes)
}

fn positive_usize(meta: &Meta, name: &str) -> Result<()> {
    let Meta::NameValue(value) = meta
    else {
        return Err(Error::new_spanned(meta, format!("`{name}` 必须写作 `{name} = <正整数>`")));
    };
    let Expr::Lit(ExprLit { lit: Lit::Int(value), .. }) = &value.value
    else {
        return Err(Error::new_spanned(&value.value, format!("`{name}` 必须是正整数")));
    };
    match value.base10_parse::<usize>() {
        Ok(0) => Err(Error::new_spanned(value, format!("`{name}` 必须大于零"))),
        Ok(_) => Ok(()),
        Err(_) => Err(Error::new_spanned(value, format!("`{name}` 必须是可表示的 usize 正整数"))),
    }
}

fn usize_value(meta: &Meta, name: &str) -> Result<()> {
    let Meta::NameValue(value) = meta
    else {
        return Err(Error::new_spanned(meta, format!("`{name}` 必须写作 `{name} = <非负整数>`")));
    };
    let Expr::Lit(ExprLit { lit: Lit::Int(value), .. }) = &value.value
    else {
        return Err(Error::new_spanned(&value.value, format!("`{name}` 必须是非负整数")));
    };
    value
        .base10_parse::<usize>()
        .map(|_| ())
        .map_err(|_| Error::new_spanned(value, format!("`{name}` 必须是可表示的 usize 非负整数")))
}

fn backend(meta: &Meta) -> Result<()> {
    let Meta::NameValue(value) = meta
    else {
        return Err(Error::new_spanned(meta, "`backend` 必须写作 `backend = Auto`"));
    };
    let Expr::Path(ExprPath { path, .. }) = &value.value
    else {
        return Err(Error::new_spanned(&value.value, "`backend` 必须是 Auto、CpuSimd、Ptx、Hip、Metal 或 Wgsl"));
    };
    let Some(segment) = path.get_ident()
    else {
        return Err(Error::new_spanned(path, "`backend` 必须是单个后端标识符"));
    };
    match segment.to_string().as_str() {
        "Auto" | "CpuSimd" | "Ptx" | "Hip" | "Metal" | "Wgsl" => Ok(()),
        _ => Err(Error::new_spanned(segment, "`backend` 必须是 Auto、CpuSimd、Ptx、Hip、Metal 或 Wgsl")),
    }
}

fn strategy(meta: &Meta) -> Result<()> {
    let Meta::NameValue(value) = meta
    else {
        return Err(Error::new_spanned(meta, "`strategy` 必须写作 `strategy = \"...\"`"));
    };
    let Expr::Lit(ExprLit { lit: Lit::Str(_), .. }) = &value.value
    else {
        return Err(Error::new_spanned(&value.value, "`strategy` 必须是字符串"));
    };
    Ok(())
}

fn validate_arguments(attributes: proc_macro2::TokenStream, declaration: Declaration) -> Result<()> {
    match declaration {
        Declaration::Neural | Declaration::Parameters => no_arguments(attributes, declaration),
        Declaration::Kernel => {
            let mut errors: Option<Error> = None;
            for meta in metas(attributes)? {
                let result = match meta.path().get_ident().map(ToString::to_string).as_deref() {
                    Some("block_size" | "vector_width" | "pipeline_depth") => {
                        positive_usize(&meta, meta.path().get_ident().unwrap().to_string().as_str())
                    }
                    Some("shared_memory_padding") => usize_value(&meta, "shared_memory_padding"),
                    Some("backend") => backend(&meta),
                    _ => Err(Error::new_spanned(
                        &meta,
                        "未知 #[kernel] 参数；允许 block_size、vector_width、pipeline_depth、shared_memory_padding、backend",
                    )),
                };
                if let Err(error) = result {
                    if let Some(all) = &mut errors {
                        all.combine(error);
                    }
                    else {
                        errors = Some(error);
                    }
                }
            }
            errors.map_or(Ok(()), Err)
        }
        Declaration::Distributed => {
            let mut errors: Option<Error> = None;
            for meta in metas(attributes)? {
                let result = match meta.path().get_ident().map(ToString::to_string).as_deref() {
                    Some("strategy") => strategy(&meta),
                    Some("world") => positive_usize(&meta, "world"),
                    _ => Err(Error::new_spanned(&meta, "未知 #[distributed] 参数；允许 strategy、world")),
                };
                if let Err(error) = result {
                    if let Some(all) = &mut errors {
                        all.combine(error);
                    }
                    else {
                        errors = Some(error);
                    }
                }
            }
            errors.map_or(Ok(()), Err)
        }
    }
}

fn expand(attributes: proc_macro2::TokenStream, item: Item, declaration: Declaration) -> TokenStream {
    if let Err(error) = validate_arguments(attributes, declaration) {
        return error.into_compile_error().into();
    }
    if !declaration.accepts(&item) {
        return Error::new_spanned(&item, format!("#[{}] 只能用于{}", declaration.name(), declaration.expected_item()))
            .into_compile_error()
            .into();
    }
    let ident = ident(&item);
    let metadata = syn::Ident::new(&format!("__TITAN_{}_META", ident), ident.span());
    let kind = declaration.name();
    quote! {
        #item
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub const #metadata: &str = concat!(#kind, ":", stringify!(#ident));
    }
    .into()
}

/// Declares a neural model structure or forward function.
#[proc_macro_attribute]
pub fn neural(attributes: TokenStream, input: TokenStream) -> TokenStream {
    expand(attributes.into(), parse_macro_input!(input as Item), Declaration::Neural)
}

/// Declares a parameter-bearing model structure.
#[proc_macro_attribute]
pub fn parameters(attributes: TokenStream, input: TokenStream) -> TokenStream {
    expand(attributes.into(), parse_macro_input!(input as Item), Declaration::Parameters)
}

/// Declares a backend-neutral kernel function and validates launch metadata.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, input: TokenStream) -> TokenStream {
    expand(attributes.into(), parse_macro_input!(input as Item), Declaration::Kernel)
}

/// Declares a distributed entrypoint and validates its declarative metadata.
#[proc_macro_attribute]
pub fn distributed(attributes: TokenStream, input: TokenStream) -> TokenStream {
    expand(attributes.into(), parse_macro_input!(input as Item), Declaration::Distributed)
}
