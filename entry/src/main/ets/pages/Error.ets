import router from '@ohos.router';

@Entry
@Component
struct Error {
  private num: number;
  private name: string;
  aboutToAppear() {
    let params = router.getParams() as Record<string, Object>;
    this.num = params.num as number;
    this.name = params.num as string;
  }

  build() {
    Column() {
      Text('error!')
      Text(this.num.toString())
      Text(this.name)
    }
  }
}