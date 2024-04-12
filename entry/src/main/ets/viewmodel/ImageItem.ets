import resourceManager from '@ohos.resourceManager';

export class ImageItem {
  private name : string;
  private format: string;
  // private fd : resourceManager.RawFileDescriptor;

  public getName() {
    return this.name
  }

  public getFormat() {
    return this.format;
  }

  constructor(name: string, format: string) {
    this.name = name
    this.format = format;
  }
}