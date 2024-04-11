import { ImageItem } from '../viewmodel/ImageItem'
import { HomeTabContentList } from '../view/HomeTabContentList'
import { HomeTabContentButton } from '../view/HomeTabContentButton'
import { CommonConstants } from '../common/constants/CommonConstants'

@Component
export struct HomeTabContent {
  @Provide imageList: Array<ImageItem> = [];

  build() {
    Column() {
      HomeTabContentList();
      HomeTabContentButton();
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}