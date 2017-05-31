// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

enum PlayingCard {
  _A(1), //
  _2(2), //
  _3(3), //
  _4(4), //
  _5(5), //
  _6(6), //
  _7(7), //
  _8(8), //
  _9(9), //
  _T(10), //
  _J(10), //
  _Q(10), //
  _K(10), //
  ;
  final int value;

  private PlayingCard(int value) {
    this.value = value;
  }
}
