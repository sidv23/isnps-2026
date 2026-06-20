-- scale-image.lua
function Image(el)
  local s = el.attributes["scale"]
  if s ~= nil then
    -- append or create style
    local style = el.attributes["style"] or ""
    if style ~= "" and not style:match(";$") then
      style = style .. ";"
    end
    style = style .. "transform: scale(" .. s .. "); transform-origin: center;"
    el.attributes["style"] = style

    -- remove custom attribute so it doesn't leak to output
    el.attributes["scale"] = nil
  end
  return el
end
