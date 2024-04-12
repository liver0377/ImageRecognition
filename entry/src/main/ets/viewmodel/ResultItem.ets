import { RecognitionType } from '../common/constants/CommonConstants'
import { ImageItem } from '../viewmodel/ImageItem'

export class ResultItem {
  private name: string;
  private recognitionType: RecognitionType | string;
  private reliability: number;
  private description: string;
  private image: ImageItem;

  constructor(
    name: string,
    recognitionType: RecognitionType | string,
    reliability: number,
    description: string,
    image: ImageItem
  ) {
    this.name = name;
    this.recognitionType = recognitionType;
    this.reliability = reliability;
    this.description = description;
    this.image = image;
  }

  // Getter 方法
  public getName(): string {
    return this.name;
  }

  public getRecognitionType(): RecognitionType | string {
    return this.recognitionType;
  }

  public getReliability(): number {
    return this.reliability;
  }

  public getDescription(): string {
    return this.description;
  }

  public getImage(): ImageItem {
    return this.image;
  }
}
