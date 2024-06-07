import { ImageItem } from '../viewmodel/ImageItem'
import { PageConstants } from '../common/constants/PageConstants'
import { HomeTabContentList } from '../view/HomeTabContentList'
import { HomeTabContentButton } from '../view/HomeTabContentButton'
import { CommonConstants } from '../common/constants/CommonConstants'
import router from '@ohos.router';
import { HomeTabContentListItem } from './HomeTabContentListItem'
import { LibraryPageListItem } from './LibraryTabContentListItem'

@Component
export struct LibraryTabContent {
  private imageList: Array<ImageItem>;
  private item: ImageItem = new ImageItem('xx', 'xx')

  build() {
    Column() {

      List({
        space: PageConstants.LIST_SPACE,
        initialIndex: PageConstants.LIST_INITIAL_INDEX
      }) {
        ListItem() {

         LibraryPageListItem({type: "动物"})



        }.onClick(() => {
          router.pushUrl({
            url : "pages/LibraryPage",
            params: {
              libraryType: "animal"
            }
          }).catch((err: Error) => {

          });
        })
        ListItem() {
          LibraryPageListItem({type: "水果"})
        }
        .onClick(() => {
          router.pushUrl({
            url : "pages/LibraryPage",
            params: {
              libraryType: "fruit"
            }
          }).catch((err: Error) => {

          });
        })
        ListItem() {
          LibraryPageListItem({type: "logo"})
        }
        .onClick(() => {
          router.pushUrl({
            url : "pages/LibraryPage",
            params: {
              libraryType: "logo"
            }
          }).catch((err: Error) => {

          });
        })
        ListItem() {
          LibraryPageListItem({type: "货币"})
        }
        .onClick(() => {
          router.pushUrl({
            url : "pages/LibraryPage",
            params: {
              libraryType: "money"
            }
          }).catch((err: Error) => {

          });
        })
        ListItem() {
          LibraryPageListItem({type: "其它"})
        }
        .onClick(() => {
          router.pushUrl({
            url : "pages/LibraryPage",
            params: {
              libraryType: "other"
            }
          }).catch((err: Error) => {

          });
        })
      }
      .backgroundColor(Color.White)
      .borderRadius($r('app.float.list_border_radius'))
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}