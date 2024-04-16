import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { CommonConstants,  RecognitionType } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import { ResultTabContent } from '../view/ResultTabContent'
import { HttpHelper } from '../common/utils/HttpUtil'
import router from '@ohos.router';
import Logger from '../common/utils/Logger'

@Entry
@Component
struct ResultPage {
  private controller: TabsController = new TabsController();
  @Provide imageName : string = "";
  @Provide httpHelper : HttpHelper = new HttpHelper("");
  @Provide currentIndex : number = 0;
  // @Provide  resultList : Array<ResultItem> = [new ResultItem("AAAA", RecognitionType.General, 100, "xxx是...", new ImageItem('image1', "jpg"))];


  // 当切换Tab之后, 需要重新识别图片
  // async generalRecognize() {
  //   let generalResponseUnitList = await this.httpHelper.generalImageRequest();
  //   let resultList : Array<ResultItem>;
  //   generalResponseUnitList.forEach((item) => {
  //     resultList.push(new ResultItem(item.getKeyword(), item.getTag(), item.getScore(), item.getDescription(), new ImageItem('image1', "jpg")))
  //   })
  //   // DEBUG
  //   Logger.info("resultList is assembled", resultList.toString());
  //
  //   return resultList
  // }

  // refresh() {
  //   // DEBUG
  //   Logger.info("Change Tab, currentIndex:", this.currentIndex.toString());
  //   if(this.currentIndex == PageConstants.TAB_BAR_FIRST)  {
  //     this.generalRecognize();
  //   }
  //   // if (this.resultList == undefined) {
  //   //   Logger.error("resultList is undefined", "");
  //   // } else if (this.resultList.length == undefined) {
  //   //   Logger.error("resultList.length is undefined resultList:", this.resultList.toString());
  //   // }
  // }

  aboutToAppear() {
    // 根据Index传递过来的参数类型将图片上传
    let params = router.getParams() as Record<string, Object>;
    this.imageName = params.imageName as string;
    // DEBUG
    if (this.imageName == null) {
      Logger.info("image is bad initialized, image:", "no!!!!!!!!!!");
    }
    // let name = this.image.getName()

    this.httpHelper = new HttpHelper(this.imageName);

    // // DEBUG
    //
    //
    // // general
    // if (this.currentIndex == PageConstants.TAB_BAR_FIRST) {
    //   this.generalRecognize()
    //     .then((resultList: Array<ResultItem>) => { this.resultList = resultList; })
    //     .catch((resultList: Array<ResultItem>) => { this.resultList = resultList});
    // }
    //
    // // DEBUG
    // if (this.resultList.length == undefined) {
    //   Logger.info("this.resultList.length is undefined", "");
    //   this.resultList = [];
    // } else  {
    //   Logger.info("this.resultList.length:    ", this.resultList.length.toString());
    //   Logger.info("resultList[0].name:     ", this.resultList[0].getName().toString())
    // }
  }

  @Builder
  TabBuilder(index: number, name: Resource) {
    Column() {
      Text(name)
        .fontColor(index == this.currentIndex ? $r('app.color.index_tab_selected_font_color') : $r('app.color.index_tab_unselected_font_color'))
        .fontSize($r('app.float.home_page_font_size'))
        .fontWeight(index == this.currentIndex ? PageConstants.FONT_WEIGHT_SELECT : PageConstants.FONT_WEIGHT_UNSELECT)
        .lineHeight(PageConstants.LINE_HEIGHT)
        .margin({
          top: PageConstants.MARGIN_TOP_TWO,
          bottom: PageConstants.MARGIN_BOTTOM
        })
      Divider()
        .strokeWidth(PageConstants.STROKE_WIDTH)
        .color($r('app.color.index_tab_selected_font_color'))
        .opacity(PageConstants.TAB_BAR_FIRST)
    }
    .width(CommonConstants.FULL_PERCENT)
    .onClick(() => {
      this.currentIndex = index;
      // this.refresh()
      this.controller.changeIndex(this.currentIndex);
    })
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.End, controller: this.controller }) {

        // general
        TabContent() {
          ResultTabContent({myIndex : 0});
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FIRST, $r('app.string.index_tab_general')))

        // animal
        TabContent() {
          ResultTabContent({myIndex : 1});
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_SECOND, $r('app.string.index_tab_animal')))

        // botany
        TabContent() {
          ResultTabContent({myIndex : 2});
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_THIRD, $r('app.string.index_tab_botany')))

        // logo
        TabContent() {
          ResultTabContent({myIndex : 3});
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FOURTH, $r('app.string.index_tab_logo')))

        // money
        TabContent() {
          ResultTabContent({ myIndex : 4});
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FIFTH, $r('app.string.index_tab_money')))

      }
      .vertical(false)
      .barMode(BarMode.Fixed)
      .barWidth(PageConstants.BAR_WIDTH)
      .barHeight(PageConstants.BAR_HEIGHT)
      .width(CommonConstants.FULL_PERCENT)
      .height(CommonConstants.FULL_PERCENT)
      .margin({ top: $r('app.float.home_tab_margin_top') })
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
    .backgroundColor($r('app.color.index_background'))
  }
}
