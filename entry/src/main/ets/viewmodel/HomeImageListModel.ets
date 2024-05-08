import { ImageItem } from './ImageItem';
import { IMAGE_DATA  } from '../common/constants/CommonConstants';
import { GlobalContext } from '../common/utils/GlobalContext';
import resourceManager from '@ohos.resourceManager';

export class HomeImageListModel {
  private imageLocalList: Array<ImageItem> = [];

  /**
   * Scan the local image.
   *
   * @return Local image list data
   */
  async getLocalImage() {
    this.imageLocalList = [];
    await this.assemblingImage();
    GlobalContext.getContext().setObject('imageLocalList', this.imageLocalList);
    return this.imageLocalList;
  }

  /**
   * Assembling the image object
   */
  async assemblingImage() {
    IMAGE_DATA.forEach(async (item: ImageItem) => {
      // let imageFd = await getContext().resourceManager.getRawFd(item.getName());
      this.imageLocalList.push(new ImageItem(item.getName(), item.getFormat()));
    });
  }


}

let homeImageListModel = new HomeImageListModel();

export default homeImageListModel as HomeImageListModel