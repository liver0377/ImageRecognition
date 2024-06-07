import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { CommonConstants, RecognitionType } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import { ResultTabContent } from '../view/ResultTabContent'
import { HttpHelper } from '../common/utils/HttpUtil'
import router from '@ohos.router';
import Logger from '../common/utils/Logger'
import { HomeTabContentListItem } from '../view/HomeTabContentListItem'
import { LibraryTabContent } from '../view/LibraryTabContent'
import { LibraryPageContentList } from '../view/LibraryPageContentList'
import { GlobalContext } from '../common/utils/GlobalContext'

@Entry
@Component
struct ResultPage {
  private controller: TabsController = new TabsController();
  private libraryType: string = "";
  private images: Array<ImageItem> = []

  aboutToAppear() {
    let params = router.getParams() as Record<string, string>;
    this.libraryType = params.libraryType as string;


    if (this.libraryType == "animal") {

      this.images = GlobalContext.getContext().getObject("动物") as Array<ImageItem>
    }
    if (this.libraryType == "fruit") {
      this.images = GlobalContext.getContext().getObject("水果") as Array<ImageItem>
    }
    if (this.libraryType == "logo") {
      this.images = GlobalContext.getContext().getObject("logo") as Array<ImageItem>
    }
    if (this.libraryType == "money") {
      this.images = GlobalContext.getContext().getObject("货币") as Array<ImageItem>
    }
    if (this.libraryType == "other") {
      this.images = GlobalContext.getContext().getObject("其它") as Array<ImageItem>
    }
  }



  build() {
    Column() {

      LibraryPageContentList({imageList: this.images})
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
    .backgroundColor($r('app.color.index_background'))
  }
}
