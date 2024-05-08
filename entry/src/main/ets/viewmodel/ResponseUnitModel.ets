export class GeneralResponseUnit {
  private keyword : string;
  private tag: string;
  private score: number;
  private description: string;
  private imageUrl: string;
  private baikeUrl: string;

  constructor(keyword: string, tag: string, score: number, description: string, imageUrl: string = "https://xxx.com", baikeUrl: string = "https://xxx.com") {
    this.keyword = keyword;
    this.tag = tag;
    this.score = score;
    this.description = description;
    this.imageUrl = imageUrl;
    this.baikeUrl = baikeUrl;
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

  getImageUrl(): string {
    return this.imageUrl;
  }

  getBaiKeUrl(): string {
    return this.baikeUrl;
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

  setImageUrl(imageUrl: string): void {
    this.imageUrl = imageUrl;
  }

  setBaikeUrl(baikeUrl: string): void {
    this.baikeUrl = baikeUrl;
  }
}


