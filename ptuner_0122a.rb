#!/usr/bin/env ruby

ps = []
1.upto(12) do |i|
  0.upto(1) do |j|
    ps << [i * 0.0001, j*0.002 + 0.001]
  end
end

total = ps.size
start = Time.now
f = open("ptuner_ruby_log", "w+")

ps.each_with_index do |p, i|
  f.puts "#{i+1}/#{total} k=16 e2=#{p[0]} e3=#{p[1]} #{(Time.now - start)/60}mins" 
  `./lambdafm -k 4 -t 15 -s 32 -e2 #{p[0]} -e3 #{p[1]} -l2 0.00002 -g k_4_e2_#{p[0]}_e3_#{p[1]} ../safe/avazu/test.libfm ../safe/avazu/train.libfm`
end

f.close



