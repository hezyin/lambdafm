#!/usr/bin/env ruby

ps = []
1.upto(8) do |i|
  1.upto(5) do |j|
    ps << [i * 0.001, j * 0.002]
  end
end

total = ps.size
start = Time.now
f = open("ptuner_ruby_log", "w+")

ps.each_with_index do |p, i|
  f.puts "#{i+1}/#{total} k=16 e2=#{p[0]} l2=#{p[1]} #{(Time.now - start)/60}mins" 
  `./lambdafm -k 16 -t 8 -s 16 -e2 #{p[0]} -e3 0.003 -l2 #{p[1]} -g k_16_e2_#{p[0]}_l2_#{p[1]} train.libfm test.libfm`
end

f.close

