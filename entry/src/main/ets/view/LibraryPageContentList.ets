import { ImageItem } from '../viewmodel/ImageItem'
import { HomeTabContentListItem } from '../view/HomeTabContentListItem'
import { HomeTabContentButton } from '../view/HomeTabContentButton'
import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import router from '@ohos.router';

@Component
export struct LibraryPageContentList {
  private imageList: Array<ImageItem>;

  build() {
    Column() {
      List({
        space: PageConstants.LIST_SPACE,
        initialIndex: PageConstants.LIST_INITIAL_INDEX
      }) {
        ForEach(this.imageList, (item: ImageItem, index?: number) => {
          ListItem() {
            HomeTabContentListItem({ item: item });
          }
        }, (item: ImageItem) => JSON.stringify(item))
      }
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}