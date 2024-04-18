import resourceManager from '@ohos.resourceManager';

export class ImageItem {
  private name : string;
  private format: string;
  private internetUrl: string;

  public getName() {
    return this.name
  }

  public getFormat() {
    return this.format;
  }

  public getInternetUrl() {
    return this.internetUrl;
  }

  constructor(name: string, format: string, internetUrl?: string) {
    this.name = name
    this.format = format;
    this.internetUrl = internetUrl;
  }
}