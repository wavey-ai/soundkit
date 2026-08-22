use soundkit::media_metadata::extract_metadata;
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), String> {
    let paths = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("usage: inspect-metadata <media-file>...".to_owned());
    }
    for path in paths {
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let metadata = extract_metadata(&bytes)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let artwork = metadata
            .artwork
            .iter()
            .map(|picture| {
                (
                    picture.picture_type,
                    picture.mime_type.clone(),
                    picture.description.clone(),
                    picture.data.len(),
                )
            })
            .collect::<Vec<_>>();
        let mut printable = metadata;
        for picture in &mut printable.artwork {
            picture.data.clear();
        }
        println!(
            "{}\n{printable:#?}\nartwork(type, mime, description, bytes)={artwork:?}",
            path.display()
        );
    }
    Ok(())
}
