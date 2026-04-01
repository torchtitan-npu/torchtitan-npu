#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SKILLS_DIR="${PROJECT_ROOT}/.agents/skills"

if [[ ! -d "${SKILLS_DIR}" ]]; then
  echo "未找到 skills 目录: ${SKILLS_DIR}"
  echo "请在项目根目录执行该脚本，或确认 .agents/skills 存在。"
  exit 1
fi

parse_frontmatter_field() {
  local file="$1"
  local key="$2"

  awk -v key="${key}" '
    BEGIN { in_frontmatter=0; separator_count=0 }
    /^---[[:space:]]*$/ {
      separator_count++
      if (separator_count == 1) {
        in_frontmatter=1
        next
      }
      if (separator_count == 2) {
        exit
      }
    }
    in_frontmatter {
      if ($0 ~ ("^" key ":[[:space:]]*")) {
        sub("^" key ":[[:space:]]*", "", $0)
        print $0
        exit
      }
    }
  ' "${file}"
}

trim_wrapped_quotes() {
  local value="$1"
  if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
    value="${value:1:${#value}-2}"
  fi
  echo "${value}"
}

show_available_skills() {
  local skill_files=()
  local file=""

  shopt -s nullglob
  skill_files=("${SKILLS_DIR}"/*/SKILL.md)
  shopt -u nullglob

  echo
  echo "可用 skills:"

  if [[ ${#skill_files[@]} -eq 0 ]]; then
    echo "- (未发现可用 skill)"
    return
  fi

  for file in "${skill_files[@]}"; do
    local skill_name
    local skill_desc

    skill_name="$(parse_frontmatter_field "${file}" "name")"
    skill_desc="$(parse_frontmatter_field "${file}" "description")"

    skill_name="$(trim_wrapped_quotes "${skill_name}")"
    skill_desc="$(trim_wrapped_quotes "${skill_desc}")"

    if [[ -z "${skill_name}" ]]; then
      skill_name="$(basename "$(dirname "${file}")")"
    fi
    if [[ -z "${skill_desc}" ]]; then
      skill_desc="(无简介)"
    fi

    echo "- ${skill_name}: ${skill_desc}"
  done
}

select_agent() {
  local choice=""

  echo "请选择当前使用的 agent:"
  echo "1) claude code (.claude)"
  echo "2) opencode (.opencode)"

  while true; do
    read -r -p "请输入序号 [1-2]: " choice
    case "${choice}" in
      1)
        AGENT_LABEL="claude code"
        AGENT_DIR_NAME=".claude"
        break
        ;;
      2)
        AGENT_LABEL="opencode"
        AGENT_DIR_NAME=".opencode"
        break
        ;;
      *)
        echo "无效输入，请输入 1 或 2。"
        ;;
    esac
  done
}

setup_agent_skills_link() {
  local agent_dir="${PROJECT_ROOT}/${AGENT_DIR_NAME}"
  local skills_link="${agent_dir}/skills"
  local relative_target="../.agents/skills"

  mkdir -p "${agent_dir}"

  if [[ -L "${skills_link}" ]]; then
    local current_target
    current_target="$(readlink "${skills_link}")"
    if [[ "${current_target}" == "${relative_target}" ]]; then
      echo "已存在软链接: ${skills_link} -> ${current_target}"
      return
    fi
  fi

  if [[ -e "${skills_link}" ]]; then
    local backup_path="${skills_link}.bak.$(date +%Y%m%d%H%M%S)"
    mv "${skills_link}" "${backup_path}"
    echo "检测到已有 ${skills_link}，已备份到 ${backup_path}"
  fi

  ln -s "${relative_target}" "${skills_link}"
  echo "已创建软链接: ${skills_link} -> ${relative_target}"
}

main() {
  echo "项目根目录: ${PROJECT_ROOT}"
  echo "skills 目录: ${SKILLS_DIR}"
  echo

  select_agent
  setup_agent_skills_link
  show_available_skills

  echo
  echo "初始化完成。你现在可以在 ${AGENT_LABEL} 中使用这些 skills。"
}

main "$@"
