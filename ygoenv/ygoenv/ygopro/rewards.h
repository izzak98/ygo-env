#ifndef YGOENV_YGOPRO_REWARDS_H_
#define YGOENV_YGOPRO_REWARDS_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ygopro
{

  struct RewardCard
  {
    std::string card_id;
    std::string location;
    std::string position;
    int seq = 0;
    bool overlay = false;
    bool negated = false;
    int level = 0;
    std::string race;
    std::vector<std::string> types;
  };

  struct NativeCondition
  {
    std::string card_id = "*";
    std::string loc;
    std::string pos;
    bool loc_partial = true;
    bool pos_partial = true;
    bool not_overlay = false;
    bool xyz_material = false;
    bool not_negated = false;
    bool has_material = false;
    std::optional<std::vector<int>> seq_in;
    std::optional<int> level_eq;
    std::optional<std::string> race_eq;
    std::optional<std::string> type_eq;
    std::optional<std::string> material_for_card_id;
  };

  struct NativeRule
  {
    std::string name;
    NativeCondition target;
    std::vector<NativeCondition> target_any_of;
    float reward = 0.f;
    bool stackable = false;
    std::vector<NativeCondition> requires_any;
    std::vector<NativeCondition> requires_all;
    std::optional<int> min_target_count;
    std::optional<std::pair<std::vector<NativeCondition>, int>> requires_min_combined;
    std::optional<std::pair<std::vector<NativeCondition>, int>> requires_exact_combined;
  };

  inline std::map<std::string, std::vector<NativeRule>> &deck_reward_rules()
  {
    static std::map<std::string, std::vector<NativeRule>> rules;
    return rules;
  }

  // ---------------------------------------------------------------------------
  // Minimal JSON (objects, arrays, strings, numbers, bool, null)
  // ---------------------------------------------------------------------------

  struct JsonVal
  {
    enum Kind
    {
      kNull,
      kBool,
      kNum,
      kStr,
      kArr,
      kObj
    } kind = kNull;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<JsonVal> a;
    std::map<std::string, JsonVal> o;

    bool is_null() const { return kind == kNull; }
    bool as_bool(bool def = false) const { return kind == kBool ? b : def; }
    double as_num(double def = 0) const { return kind == kNum ? n : def; }
    const std::string &as_str() const
    {
      static const std::string empty;
      return kind == kStr ? s : empty;
    }
    const JsonVal *get(const char *key) const
    {
      if (kind != kObj)
      {
        return nullptr;
      }
      auto it = o.find(key);
      return it == o.end() ? nullptr : &it->second;
    }
  };

  class JsonParser
  {
  public:
    explicit JsonParser(const std::string &in) : s_(in), i_(0) {}

    JsonVal parse()
    {
      skip();
      JsonVal v = parse_value();
      skip();
      return v;
    }

  private:
    const std::string &s_;
    size_t i_;

    void skip()
    {
      while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_])))
      {
        ++i_;
      }
    }

    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }

    char getc() { return i_ < s_.size() ? s_[i_++] : '\0'; }

    JsonVal parse_value()
    {
      skip();
      char c = peek();
      if (c == '{')
      {
        return parse_object();
      }
      if (c == '[')
      {
        return parse_array();
      }
      if (c == '"')
      {
        return parse_string();
      }
      if (c == 't' || c == 'f')
      {
        return parse_bool();
      }
      if (c == 'n')
      {
        return parse_null();
      }
      return parse_number();
    }

    JsonVal parse_object()
    {
      JsonVal v;
      v.kind = JsonVal::kObj;
      getc(); // {
      skip();
      if (peek() == '}')
      {
        getc();
        return v;
      }
      while (true)
      {
        skip();
        JsonVal key = parse_string();
        skip();
        if (getc() != ':')
        {
          throw std::runtime_error("JSON expected ':'");
        }
        v.o[key.s] = parse_value();
        skip();
        char c = getc();
        if (c == '}')
        {
          break;
        }
        if (c != ',')
        {
          throw std::runtime_error("JSON expected ',' or '}'");
        }
      }
      return v;
    }

    JsonVal parse_array()
    {
      JsonVal v;
      v.kind = JsonVal::kArr;
      getc(); // [
      skip();
      if (peek() == ']')
      {
        getc();
        return v;
      }
      while (true)
      {
        v.a.push_back(parse_value());
        skip();
        char c = getc();
        if (c == ']')
        {
          break;
        }
        if (c != ',')
        {
          throw std::runtime_error("JSON expected ',' or ']'");
        }
      }
      return v;
    }

    JsonVal parse_string()
    {
      JsonVal v;
      v.kind = JsonVal::kStr;
      if (getc() != '"')
      {
        throw std::runtime_error("JSON expected string");
      }
      while (i_ < s_.size())
      {
        char c = getc();
        if (c == '"')
        {
          break;
        }
        if (c == '\\')
        {
          char e = getc();
          switch (e)
          {
          case '"':
          case '\\':
          case '/':
            v.s.push_back(e);
            break;
          case 'b':
            v.s.push_back('\b');
            break;
          case 'f':
            v.s.push_back('\f');
            break;
          case 'n':
            v.s.push_back('\n');
            break;
          case 'r':
            v.s.push_back('\r');
            break;
          case 't':
            v.s.push_back('\t');
            break;
          case 'u':
          {
            unsigned code = 0;
            for (int k = 0; k < 4; ++k)
            {
              char h = getc();
              code <<= 4;
              if (h >= '0' && h <= '9')
              {
                code += static_cast<unsigned>(h - '0');
              }
              else if (h >= 'a' && h <= 'f')
              {
                code += static_cast<unsigned>(h - 'a' + 10);
              }
              else if (h >= 'A' && h <= 'F')
              {
                code += static_cast<unsigned>(h - 'A' + 10);
              }
            }
            if (code < 0x80)
            {
              v.s.push_back(static_cast<char>(code));
            }
            else if (code < 0x800)
            {
              v.s.push_back(static_cast<char>(0xC0 | (code >> 6)));
              v.s.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            else
            {
              v.s.push_back(static_cast<char>(0xE0 | (code >> 12)));
              v.s.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              v.s.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default:
            v.s.push_back(e);
            break;
          }
        }
        else
        {
          v.s.push_back(c);
        }
      }
      return v;
    }

    JsonVal parse_bool()
    {
      JsonVal v;
      v.kind = JsonVal::kBool;
      if (s_.compare(i_, 4, "true") == 0)
      {
        i_ += 4;
        v.b = true;
      }
      else if (s_.compare(i_, 5, "false") == 0)
      {
        i_ += 5;
        v.b = false;
      }
      else
      {
        throw std::runtime_error("JSON invalid bool");
      }
      return v;
    }

    JsonVal parse_null()
    {
      if (s_.compare(i_, 4, "null") != 0)
      {
        throw std::runtime_error("JSON invalid null");
      }
      i_ += 4;
      return JsonVal{};
    }

    JsonVal parse_number()
    {
      JsonVal v;
      v.kind = JsonVal::kNum;
      size_t start = i_;
      if (peek() == '-')
      {
        getc();
      }
      while (std::isdigit(static_cast<unsigned char>(peek())))
      {
        getc();
      }
      if (peek() == '.')
      {
        getc();
        while (std::isdigit(static_cast<unsigned char>(peek())))
        {
          getc();
        }
      }
      if (peek() == 'e' || peek() == 'E')
      {
        getc();
        if (peek() == '+' || peek() == '-')
        {
          getc();
        }
        while (std::isdigit(static_cast<unsigned char>(peek())))
        {
          getc();
        }
      }
      v.n = std::stod(s_.substr(start, i_ - start));
      return v;
    }
  };

  inline bool ieq(const std::string &a, const std::string &b)
  {
    if (a.size() != b.size())
    {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
      if (std::tolower(static_cast<unsigned char>(a[i])) !=
          std::tolower(static_cast<unsigned char>(b[i])))
      {
        return false;
      }
    }
    return true;
  }

  inline NativeCondition condition_from_json(const JsonVal &j)
  {
    NativeCondition c;
    if (const JsonVal *id = j.get("id"))
    {
      c.card_id = id->as_str();
    }
    if (const JsonVal *pos = j.get("pos"))
    {
      if (const JsonVal *t = pos->get("type"))
      {
        c.pos = t->as_str();
      }
      if (const JsonVal *m = pos->get("match"))
      {
        c.pos_partial = m->as_str() != "full";
      }
    }
    if (const JsonVal *loc = j.get("loc"))
    {
      if (const JsonVal *t = loc->get("type"))
      {
        c.loc = t->as_str();
      }
      if (const JsonVal *m = loc->get("match"))
      {
        c.loc_partial = m->as_str() != "full";
      }
    }
    if (const JsonVal *fr = j.get("further_restrictions"))
    {
      if (const JsonVal *v = fr->get("overlay"))
      {
        c.not_overlay = v->as_bool();
      }
      if (const JsonVal *v = fr->get("xyz_material"))
      {
        c.xyz_material = v->as_bool();
      }
      if (const JsonVal *v = fr->get("not_negated"))
      {
        c.not_negated = v->as_bool();
      }
      if (const JsonVal *v = fr->get("has_material"))
      {
        c.has_material = v->as_bool();
      }
      if (const JsonVal *v = fr->get("seq_in"))
      {
        std::vector<int> seqs;
        for (const auto &x : v->a)
        {
          seqs.push_back(static_cast<int>(x.as_num()));
        }
        c.seq_in = seqs;
      }
      if (const JsonVal *v = fr->get("level_eq"))
      {
        c.level_eq = static_cast<int>(v->as_num());
      }
      if (const JsonVal *v = fr->get("race_eq"))
      {
        c.race_eq = v->as_str();
      }
      if (const JsonVal *v = fr->get("type_eq"))
      {
        c.type_eq = v->as_str();
      }
      if (const JsonVal *v = fr->get("material_for_card_id"))
      {
        c.material_for_card_id = v->as_str();
      }
    }
    return c;
  }

  inline NativeRule rule_from_json(const JsonVal &j)
  {
    NativeRule r;
    r.target = condition_from_json(j);
    r.reward = static_cast<float>(j.get("reward") ? j.get("reward")->as_num() : 0);
    r.stackable = j.get("stackable") && j.get("stackable")->as_bool();
    if (const JsonVal *n = j.get("name"))
    {
      r.name = n->as_str();
    }
    if (const JsonVal *fc = j.get("further_conditions"))
    {
      std::string logic = fc->get("logic") ? fc->get("logic")->as_str() : "";
      const JsonVal *conds = fc->get("conditions");
      if (conds)
      {
        for (const auto &cj : conds->a)
        {
          NativeCondition c = condition_from_json(cj);
          if (logic == "OR")
          {
            r.requires_any.push_back(c);
          }
          else if (logic == "AND")
          {
            r.requires_all.push_back(c);
          }
        }
      }
    }
    if (const JsonVal *mt = j.get("min_target_count"))
    {
      r.min_target_count = static_cast<int>(mt->as_num());
    }
    if (const JsonVal *mc = j.get("min_combined_count"))
    {
      std::vector<NativeCondition> conds;
      if (const JsonVal *cs = mc->get("conditions"))
      {
        for (const auto &cj : cs->a)
        {
          conds.push_back(condition_from_json(cj));
        }
      }
      int mn = mc->get("min") ? static_cast<int>(mc->get("min")->as_num()) : 0;
      r.requires_min_combined = std::make_pair(std::move(conds), mn);
    }
    if (const JsonVal *ec = j.get("exact_combined_count"))
    {
      std::vector<NativeCondition> conds;
      if (const JsonVal *cs = ec->get("conditions"))
      {
        for (const auto &cj : cs->a)
        {
          conds.push_back(condition_from_json(cj));
        }
      }
      int ex = ec->get("exact") ? static_cast<int>(ec->get("exact")->as_num()) : 0;
      r.requires_exact_combined = std::make_pair(std::move(conds), ex);
    }
    if (const JsonVal *tao = j.get("target_any_of"))
    {
      for (const auto &cj : tao->a)
      {
        r.target_any_of.push_back(condition_from_json(cj));
      }
    }
    return r;
  }

  inline void load_reward_json(const std::string &json)
  {
    if (json.empty())
    {
      return;
    }
    JsonParser p(json);
    JsonVal root = p.parse();
    auto &dst = deck_reward_rules();
    dst.clear();
    for (const auto &[deck, rules_j] : root.o)
    {
      std::vector<NativeRule> rules;
      for (const auto &rj : rules_j.a)
      {
        rules.push_back(rule_from_json(rj));
      }
      dst[deck] = std::move(rules);
    }
  }

  inline bool matches_string(const std::string &actual, const std::string &expected,
                             bool partial)
  {
    if (expected.empty())
    {
      return true;
    }
    if (actual.empty())
    {
      return false;
    }
    if (!partial)
    {
      return actual == expected;
    }
    return actual.find(expected) != std::string::npos;
  }

  inline bool has_materials(const std::vector<RewardCard> &cards, int seq,
                            const std::string &location)
  {
    for (const auto &c : cards)
    {
      if (c.overlay && c.seq == seq && c.location == location)
      {
        return true;
      }
    }
    return false;
  }

  inline int check_condition(const std::vector<RewardCard> &cards,
                             const NativeCondition &cond)
  {
    int count = 0;
    for (const auto &card : cards)
    {
      if (cond.card_id != "*" && card.card_id != cond.card_id)
      {
        continue;
      }
      if (!matches_string(card.position, cond.pos, cond.pos_partial))
      {
        continue;
      }
      if (!matches_string(card.location, cond.loc, cond.loc_partial))
      {
        continue;
      }
      if (cond.not_overlay && card.overlay)
      {
        continue;
      }
      if (cond.xyz_material && !card.overlay)
      {
        continue;
      }
      if (cond.material_for_card_id)
      {
        if (!card.overlay)
        {
          continue;
        }
        bool attached = false;
        const std::string &host_id = *cond.material_for_card_id;
        for (const auto &h : cards)
        {
          if (h.overlay)
          {
            continue;
          }
          if (h.card_id != host_id)
          {
            continue;
          }
          if (h.seq != card.seq)
          {
            continue;
          }
          if (card.location.empty() || h.location != card.location)
          {
            continue;
          }
          attached = true;
          break;
        }
        if (!attached)
        {
          continue;
        }
      }
      if (cond.seq_in)
      {
        bool ok = false;
        for (int s : *cond.seq_in)
        {
          if (card.seq == s)
          {
            ok = true;
            break;
          }
        }
        if (!ok)
        {
          continue;
        }
      }
      if (cond.level_eq && card.level != *cond.level_eq)
      {
        continue;
      }
      if (cond.race_eq && !ieq(card.race, *cond.race_eq))
      {
        continue;
      }
      if (cond.type_eq)
      {
        bool ok = false;
        for (const auto &t : card.types)
        {
          if (ieq(t, *cond.type_eq))
          {
            ok = true;
            break;
          }
        }
        if (!ok)
        {
          continue;
        }
      }
      if (cond.has_material && !has_materials(cards, card.seq, card.location))
      {
        continue;
      }
      if (cond.not_negated && card.negated)
      {
        continue;
      }
      count++;
    }
    return count;
  }

  inline float evaluate_rule(const std::vector<RewardCard> &cards, const NativeRule &rule)
  {
    int target_count = 0;
    if (!rule.target_any_of.empty())
    {
      for (const auto &c : rule.target_any_of)
      {
        target_count = std::max(target_count, check_condition(cards, c));
      }
      if (target_count == 0)
      {
        return 0.f;
      }
    }
    else
    {
      target_count = check_condition(cards, rule.target);
      if (target_count == 0)
      {
        return 0.f;
      }
    }
    if (rule.min_target_count && target_count < *rule.min_target_count)
    {
      return 0.f;
    }
    if (!rule.requires_any.empty())
    {
      bool any_met = false;
      for (const auto &c : rule.requires_any)
      {
        if (check_condition(cards, c) > 0)
        {
          any_met = true;
          break;
        }
      }
      if (!any_met)
      {
        return 0.f;
      }
    }
    else if (!rule.requires_all.empty())
    {
      for (const auto &c : rule.requires_all)
      {
        if (check_condition(cards, c) == 0)
        {
          return 0.f;
        }
      }
    }
    if (rule.requires_min_combined)
    {
      int combined = 0;
      for (const auto &c : rule.requires_min_combined->first)
      {
        combined += check_condition(cards, c);
      }
      if (combined < rule.requires_min_combined->second)
      {
        return 0.f;
      }
    }
    if (rule.requires_exact_combined)
    {
      int combined = 0;
      for (const auto &c : rule.requires_exact_combined->first)
      {
        combined += check_condition(cards, c);
      }
      if (combined != rule.requires_exact_combined->second)
      {
        return 0.f;
      }
    }
    if (rule.stackable)
    {
      return rule.reward * static_cast<float>(target_count);
    }
    return rule.reward;
  }

  inline float evaluate_deck_reward(const std::string &deck_name,
                                    const std::vector<RewardCard> &cards)
  {
    auto &all = deck_reward_rules();
    auto it = all.find(deck_name);
    if (it == all.end())
    {
      return 0.f;
    }
    float total = 0.f;
    for (const auto &rule : it->second)
    {
      total += evaluate_rule(cards, rule);
    }
    return total;
  }

  // Returns per-rule (name, value) contributions for the given deck and cards.
  inline std::vector<std::pair<std::string, float>> evaluate_deck_reward_breakdown(
      const std::string &deck_name,
      const std::vector<RewardCard> &cards)
  {
    std::vector<std::pair<std::string, float>> breakdown;
    auto &all = deck_reward_rules();
    auto it = all.find(deck_name);
    if (it == all.end())
    {
      return breakdown;
    }
    for (const auto &rule : it->second)
    {
      float val = evaluate_rule(cards, rule);
      breakdown.emplace_back(rule.name, val);
    }
    return breakdown;
  }

} // namespace ygopro

#endif
