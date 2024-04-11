import resourceManager from '@ohos.resourceManager';

export class ImageItem {
  private name : string;
  private fd : resourceManager.RawFileDescriptor;

  public getName() {
    return this.name
  }

  public getFd() {
    return this.fd
  }

  constructor(name: string, fd : resourceManager.RawFileDescriptor) {
    this.name = name
    this.fd = fd
  }
}