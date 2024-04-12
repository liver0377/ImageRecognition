export class GeneralResponseUnit {
  private keyword : string;
  private tag: string;
  private score: number;
  private description: string;

  constructor(keyword: string, tag: string, score: number, description: string) {
    this.keyword = keyword;
    this.tag = tag;
    this.score = score;
    this.description = description;
  }

  // Getter methods
  getKeyword(): string {
    return this.keyword;
  }

  getTag(): string {
    return this.tag;
  }

  getScore(): number {
    return this.score;
  }

  getDescription(): string {
    return this.description;
  }

  // Setter methods
  setKeyword(keyword: string): void {
    this.keyword = keyword;
  }

  setTag(tag: string): void {
    this.tag = tag;
  }

  setScore(score: number): void {
    this.score = score;
  }

  setDescription(description: string): void {
    this.description = description;
  }
}


