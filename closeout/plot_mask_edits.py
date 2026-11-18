"""Regenerate the public SVG from the aggregate JSON; no model or images needed."""
import json
from pathlib import Path


def render(data):
    left = 215
    scale = 4.6
    rows = [("lesion_recovered", "Lesion pixels recovered"), ("lesion_erased", "Lesion pixels erased"),
            ("false_positive_removed", "False positives removed"), ("false_positive_added", "False positives added")]
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 400" role="img" aria-labelledby="title desc">',
                '<title id="title">Pixel changes on entropy-selected retries</title>',
                '<desc id="desc">Dots are per-scan medians and whiskers span the middle 50 percent. '
                'Worsened scans had a median of 29.0 percent of lesion area in added false positives, compared with 3.9 percent on improved scans.</desc>',
                '<rect width="860" height="400" fill="white"/>',
                '<g font-family="sans-serif" fill="#1d303d">',
                '<text x="18" y="34" font-size="22" font-weight="bold">What changed when a mask was retried?</text>',
                '<text x="18" y="58" font-size="14">Edited pixels as a percentage of reference lesion area · BUS-BRA fold 4 · post hoc</text>']
    for tick in (0,25,50,75,100):
        pos = left+scale*tick
        elements += [f'<path d="M{pos} 106V325" stroke="#d4dce1"/>',
                     f'<text x="{pos}" y="346" text-anchor="middle" font-size="13">{tick}%</text>']
    for index,(name,color,marker) in enumerate((("improved","#007367","circle"),("worsened","#973e27","square"))):
        col = 18+index*222
        label = f'{name.capitalize()} ({data[name]["scans"]} scans)'
        def mark(pos, height):
            if marker=="circle":
                return f'<circle cx="{pos}" cy="{height}" r="4.5" fill="{color}"/>'
            return f'<rect x="{pos-4}" y="{height-4}" width="8" height="8" fill="{color}"/>'
        elements += [mark(col+5,85),f'<text x="{col+18}" y="90" font-size="14">{label}</text>']
        for number,(key,title) in enumerate(rows):
            height = 130+number*55+(index*12-6)
            if index==0:
                elements.append(f'<text x="18" y="{height+11}" font-size="15">{title}</text>')
            low,mid,high = [left+scale*100*value for value in data[name]["quantiles"][key]]
            elements += [f'<path d="M{low} {height}H{high} M{low} {height-3}v6 M{high} {height-3}v6" stroke="{color}" stroke-width="2"/>',
                         mark(mid,height)]
    elements += ['<text x="18" y="379" font-size="13">Dots/squares: medians. Whiskers: interquartile ranges. One unchanged scan is not plotted.</text>',
                 '</g></svg>']
    return "\n".join(elements)+"\n"


if __name__=="__main__":
    folder = Path(__file__).parent
    (folder/"mask-edits.svg").write_text(render(json.loads((folder/"mask-audit.aggregate.json").read_text())))
