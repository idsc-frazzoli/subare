// code by fluric
package ch.ethz.idsc.subare.util;

import java.util.Random;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Clip;

public class Coinflip {
  /** @param p_head
   * @return new instance of Coinflip with given probability p_head that tossHead() returns true */
  public static Coinflip of(Scalar p_head) {
    return new Coinflip(p_head);
  }

  /** Quote from Wikipedia:
   * "a fair coin is an idealized randomizing device with two states
   * (usually named "heads" and "tails") which are equally likely to occur."
   * 
   * @return new instance of Coinflip for which tossHead() returns true with probability 1/2 */
  public static Coinflip fair() {
    return of(RationalScalar.HALF);
  }

  // ---
  private final Random random = new Random();
  private final float p_head;

  private Coinflip(Scalar p_head) {
    this.p_head = Clip.unit().requireInside(p_head).number().floatValue();
  }

  /** returns true if the coin toss ended up with head */
  public boolean tossHead() {
    return random.nextFloat() < p_head;
  }
}
