//! Task compiler: natural-language intent → TaskProfile.
//!
//! Heuristic keyword compiler for Phase 1. Replace with a learned classifier later.

use crate::types::{TaskCategory, TaskProfile};

pub struct TaskCompiler {
    default_memory_limit_bytes: u64,
}

impl Default for TaskCompiler {
    fn default() -> Self {
        Self {
            default_memory_limit_bytes: 1024 * 1024 * 1024, // 1 GiB
        }
    }
}

impl TaskCompiler {
    pub fn new(default_memory_limit_bytes: u64) -> Self {
        Self {
            default_memory_limit_bytes,
        }
    }

    pub fn compile(&self, prompt: &str) -> TaskProfile {
        let lower = prompt.to_lowercase();
        let domain = detect_domain(&lower);
        let language = detect_language(&lower);
        let mut skills = detect_skills(domain, &lower, language.as_deref());

        if skills.is_empty() {
            skills.push(domain.as_str().to_string());
        }

        TaskProfile {
            domain,
            skills,
            language,
            memory_limit_bytes: self.default_memory_limit_bytes,
            raw_prompt: prompt.trim().to_string(),
        }
    }

    pub fn compile_category(
        &self,
        category: TaskCategory,
        language: Option<String>,
        memory_limit_bytes: Option<u64>,
    ) -> TaskProfile {
        let skills = default_skills(category, language.as_deref());
        TaskProfile {
            domain: category,
            skills,
            language,
            memory_limit_bytes: memory_limit_bytes.unwrap_or(self.default_memory_limit_bytes),
            raw_prompt: format!("{} specialist", category.as_str()),
        }
    }
}

fn detect_domain(text: &str) -> TaskCategory {
    if contains_any(
        text,
        &[
            "code",
            "coding",
            "program",
            "rust",
            "python",
            "javascript",
            "debug",
        ],
    ) {
        TaskCategory::Coding
    } else if contains_any(text, &["math", "algebra", "calculus", "proof", "equation"]) {
        TaskCategory::Math
    } else if contains_any(text, &["writ", "essay", "blog", "copy", "story"]) {
        TaskCategory::Writing
    } else if contains_any(text, &["research", "paper", "literature", "citation"]) {
        TaskCategory::Research
    } else if contains_any(text, &["medical", "clinical", "patient", "diagnosis"]) {
        TaskCategory::Medical
    } else {
        TaskCategory::Custom
    }
}

fn detect_language(text: &str) -> Option<String> {
    const LANGS: &[(&str, &str)] = &[
        ("python", "python"),
        ("rust", "rust"),
        ("javascript", "javascript"),
        ("typescript", "typescript"),
        ("java", "java"),
        ("go ", "go"),
        ("c++", "cpp"),
    ];
    for (needle, lang) in LANGS {
        if text.contains(needle) {
            return Some((*lang).to_string());
        }
    }
    None
}

fn detect_skills(domain: TaskCategory, text: &str, language: Option<&str>) -> Vec<String> {
    let mut skills = default_skills(domain, language);
    if text.contains("debug") {
        push_unique(&mut skills, "debugging");
    }
    if text.contains("architect") {
        push_unique(&mut skills, "software architecture");
    }
    if text.contains("test") {
        push_unique(&mut skills, "testing");
    }
    skills
}

fn default_skills(domain: TaskCategory, language: Option<&str>) -> Vec<String> {
    let mut skills = match domain {
        TaskCategory::Coding => vec!["debugging".into(), "software architecture".into()],
        TaskCategory::Math => vec!["algebra".into(), "reasoning".into()],
        TaskCategory::Writing => vec!["editing".into(), "clarity".into()],
        TaskCategory::Research => vec!["literature review".into(), "summarization".into()],
        TaskCategory::Medical => vec!["medical text".into(), "research".into()],
        TaskCategory::Custom => vec!["general".into()],
    };
    if let Some(lang) = language {
        skills.insert(0, lang.to_string());
    }
    skills
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| text.contains(n))
}

fn push_unique(skills: &mut Vec<String>, skill: &str) {
    if !skills.iter().any(|s| s == skill) {
        skills.push(skill.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_python_coding_assistant() {
        let compiler = TaskCompiler::default();
        let profile = compiler.compile("I need a Python coding assistant");
        assert_eq!(profile.domain, TaskCategory::Coding);
        assert_eq!(profile.language.as_deref(), Some("python"));
        assert!(profile.skills.iter().any(|s| s == "python"));
    }
}
