export function truncateChineseCharacters(str, maxLength) {
  let len = 0;
  let result = '';
  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    if (/[\\u4e00-\\u9fa5]/.test(char)) {
      len += 2; // 中文字符长度为两个字节
    } else {
      len += 1; // 非中文字符长度为一个字节
    }
    if (len <= maxLength) {
      result += char;
    } else {
      break;
    }
  }
  return result;
}