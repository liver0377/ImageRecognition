from urllib.parse import parse_qs


def parse_request_body(request_body: str):
    parsed_body = parse_qs(request_body)
    imageCode = parsed_body.get("image", [])[0]
    top_num = int(parsed_body.get("top_num", [5])[0])
    baike_num = int(parsed_body.get("baike_num", [5])[0])
    return imageCode, top_num, baike_num
